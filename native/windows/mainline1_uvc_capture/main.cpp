#include <windows.h>
#include <cfgmgr32.h>
#include <bcrypt.h>
#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <setupapi.h>
#include <fcntl.h>
#include <io.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cwctype>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

void Check(HRESULT value, const char* label) {
  if (FAILED(value)) {
    char message[160]{};
    std::snprintf(message, sizeof(message), "%s failed: 0x%08lx", label,
                  static_cast<unsigned long>(value));
    throw std::runtime_error(message);
  }
}

std::wstring Lower(std::wstring value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
  return value;
}

std::string Utf8(const std::wstring& value) {
  if (value.empty()) return {};
  const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                       static_cast<int>(value.size()), nullptr, 0,
                                       nullptr, nullptr);
  if (size <= 0) throw std::runtime_error("UTF-8 conversion failed");
  std::string output(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      output.data(), size, nullptr, nullptr);
  return output;
}

std::string JsonEscape(const std::string& value) {
  std::string output;
  for (const unsigned char c : value) {
    if (c == '\\' || c == '"') {
      output.push_back('\\');
      output.push_back(static_cast<char>(c));
    } else if (c >= 0x20) {
      output.push_back(static_cast<char>(c));
    }
  }
  return output;
}

void PutBe32(std::vector<uint8_t>& output, uint32_t value) {
  output.push_back(static_cast<uint8_t>(value >> 24));
  output.push_back(static_cast<uint8_t>(value >> 16));
  output.push_back(static_cast<uint8_t>(value >> 8));
  output.push_back(static_cast<uint8_t>(value));
}

void PutBe64(std::vector<uint8_t>& output, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void WriteAll(const uint8_t* data, size_t size) {
  while (size > 0) {
    const size_t written = std::fwrite(data, 1, size, stdout);
    if (written == 0) throw std::runtime_error("stdout write failed");
    data += written;
    size -= written;
  }
  std::fflush(stdout);
}

std::vector<uint8_t> Sha256(const uint8_t* data, size_t size) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0;
  DWORD copied = 0;
  Check(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
        "BCryptOpenAlgorithmProvider");
  Check(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                          &copied, 0), "BCryptGetProperty");
  std::vector<uint8_t> object(object_bytes);
  std::vector<uint8_t> digest(32);
  Check(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr, 0, 0),
        "BCryptCreateHash");
  Check(BCryptHashData(hash, const_cast<PUCHAR>(data), static_cast<ULONG>(size), 0),
        "BCryptHashData");
  Check(BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0),
        "BCryptFinishHash");
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  return digest;
}

struct Args {
  std::wstring locator;
  std::string fingerprint;
  uint32_t width = 1920;
  uint32_t height = 1080;
  uint32_t fps = 60;
  std::vector<std::pair<std::wstring, long>> controls;
};

Args ParseArgs(int argc, wchar_t** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::wstring name = argv[index];
    if (index + 1 >= argc) throw std::runtime_error("missing argument value");
    const std::wstring value = argv[++index];
    if (name == L"--device-locator") args.locator = value;
    else if (name == L"--fingerprint") args.fingerprint = Utf8(value);
    else if (name == L"--width") args.width = static_cast<uint32_t>(std::stoul(value));
    else if (name == L"--height") args.height = static_cast<uint32_t>(std::stoul(value));
    else if (name == L"--fps") args.fps = static_cast<uint32_t>(std::stoul(value));
    else if (name == L"--pixel-format" && value != L"MJPG")
      throw std::runtime_error("only MJPG is supported");
    else if (name == L"--control") {
      const size_t equal = value.find(L'=');
      if (equal == std::wstring::npos) throw std::runtime_error("invalid control");
      args.controls.emplace_back(value.substr(0, equal),
                                 std::stol(value.substr(equal + 1)));
    } else if (name != L"--pixel-format") {
      throw std::runtime_error("unknown argument");
    }
  }
  if (args.locator.empty() || args.fingerprint.size() != 64)
    throw std::runtime_error("identity arguments are required");
  return args;
}

std::wstring InstanceForInterface(const std::wstring& target) {
  HDEVINFO devices = SetupDiGetClassDevsW(&KSCATEGORY_VIDEO_CAMERA, nullptr, nullptr,
      DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
  if (devices == INVALID_HANDLE_VALUE) throw std::runtime_error("SetupDiGetClassDevs failed");
  std::wstring found;
  for (DWORD index = 0;; ++index) {
    SP_DEVICE_INTERFACE_DATA interface_data{};
    interface_data.cbSize = sizeof(interface_data);
    if (!SetupDiEnumDeviceInterfaces(devices, nullptr, &KSCATEGORY_VIDEO_CAMERA,
                                     index, &interface_data)) {
      if (GetLastError() == ERROR_NO_MORE_ITEMS) break;
      SetupDiDestroyDeviceInfoList(devices);
      throw std::runtime_error("SetupDiEnumDeviceInterfaces failed");
    }
    DWORD bytes = 0;
    SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, nullptr, 0, &bytes, nullptr);
    std::vector<uint8_t> storage(bytes);
    auto* detail = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA_W*>(storage.data());
    detail->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_W);
    SP_DEVINFO_DATA device_data{};
    device_data.cbSize = sizeof(device_data);
    if (!SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, detail, bytes,
                                          nullptr, &device_data)) continue;
    if (Lower(detail->DevicePath) != Lower(target)) continue;
    wchar_t instance[MAX_DEVICE_ID_LEN]{};
    if (!SetupDiGetDeviceInstanceIdW(devices, &device_data, instance,
                                     MAX_DEVICE_ID_LEN, nullptr)) {
      SetupDiDestroyDeviceInfoList(devices);
      throw std::runtime_error("SetupDiGetDeviceInstanceId failed");
    }
    found = instance;
    break;
  }
  SetupDiDestroyDeviceInfoList(devices);
  return found;
}

struct SelectedSource {
  ComPtr<IMFActivate> activation;
  std::wstring symbolic_link;
};

SelectedSource SelectSource(const std::wstring& locator) {
  ComPtr<IMFAttributes> attributes;
  Check(MFCreateAttributes(&attributes, 1), "MFCreateAttributes");
  Check(attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
      MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID), "SetGUID");
  IMFActivate** raw = nullptr;
  UINT32 count = 0;
  Check(MFEnumDeviceSources(attributes.Get(), &raw, &count), "MFEnumDeviceSources");
  SelectedSource selected;
  for (UINT32 index = 0; index < count; ++index) {
    UINT32 length = 0;
    wchar_t* link = nullptr;
    if (SUCCEEDED(raw[index]->GetAllocatedString(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &link, &length))) {
      const std::wstring symbolic(link, length);
      CoTaskMemFree(link);
      if (Lower(InstanceForInterface(symbolic)) == Lower(locator)) {
        if (selected.activation) throw std::runtime_error("PnP locator mapped twice");
        selected.activation = raw[index];
        selected.symbolic_link = symbolic;
      }
    }
  }
  for (UINT32 index = 0; index < count; ++index) raw[index]->Release();
  CoTaskMemFree(raw);
  if (!selected.activation) throw std::runtime_error("PnP locator has no MF capture source");
  return selected;
}

bool LockControls(IMFMediaSource* source,
                  const std::vector<std::pair<std::wstring, long>>& controls) {
  ComPtr<IAMCameraControl> camera;
  ComPtr<IAMVideoProcAmp> proc;
  source.As(&camera);
  source.As(&proc);
  const std::map<std::wstring, long> camera_ids = {
      {L"exposure", CameraControl_Exposure}, {L"focus", CameraControl_Focus},
      {L"zoom", CameraControl_Zoom}};
  const std::map<std::wstring, long> proc_ids = {
      {L"white_balance_temperature", VideoProcAmp_WhiteBalance},
      {L"gain", VideoProcAmp_Gain},
      {L"brightness", VideoProcAmp_Brightness}, {L"contrast", VideoProcAmp_Contrast},
      {L"saturation", VideoProcAmp_Saturation}, {L"sharpness", VideoProcAmp_Sharpness}};
  for (const auto& [name, value] : controls) {
    long actual = 0;
    long flags = 0;
    if (name == L"auto_focus" || name == L"auto_exposure") {
      if (!camera) return false;
      const long property = name == L"auto_focus" ? CameraControl_Focus
                                                   : CameraControl_Exposure;
      if (FAILED(camera->Get(property, &actual, &flags))) return false;
      const long requested_flags = value == 0 ? CameraControl_Flags_Manual
                                               : CameraControl_Flags_Auto;
      if (FAILED(camera->Set(property, actual, requested_flags)) ||
          FAILED(camera->Get(property, &actual, &flags)) ||
          flags != requested_flags) return false;
      continue;
    }
    if (name == L"auto_white_balance") {
      if (!proc || FAILED(proc->Get(VideoProcAmp_WhiteBalance, &actual, &flags)))
        return false;
      const long requested_flags = value == 0 ? VideoProcAmp_Flags_Manual
                                               : VideoProcAmp_Flags_Auto;
      if (FAILED(proc->Set(VideoProcAmp_WhiteBalance, actual, requested_flags)) ||
          FAILED(proc->Get(VideoProcAmp_WhiteBalance, &actual, &flags)) ||
          flags != requested_flags) return false;
      continue;
    }
    if (camera_ids.contains(name)) {
      if (!camera || FAILED(camera->Set(camera_ids.at(name), value,
                                        CameraControl_Flags_Manual)) ||
          FAILED(camera->Get(camera_ids.at(name), &actual, &flags)) || actual != value ||
          flags != CameraControl_Flags_Manual) return false;
    } else if (proc_ids.contains(name)) {
      if (!proc || FAILED(proc->Set(proc_ids.at(name), value,
                                    VideoProcAmp_Flags_Manual)) ||
          FAILED(proc->Get(proc_ids.at(name), &actual, &flags)) || actual != value ||
          flags != VideoProcAmp_Flags_Manual) return false;
    } else {
      return false;
    }
  }
  return !controls.empty();
}

void SelectNativeMode(IMFSourceReader* reader, const Args& args) {
  for (DWORD index = 0;; ++index) {
    ComPtr<IMFMediaType> type;
    const HRESULT status = reader->GetNativeMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM, index, &type);
    if (status == MF_E_NO_MORE_TYPES) break;
    Check(status, "GetNativeMediaType");
    GUID subtype{};
    UINT32 width = 0, height = 0, numerator = 0, denominator = 0;
    if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype)) || subtype != MFVideoFormat_MJPG ||
        FAILED(MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &width, &height)) ||
        FAILED(MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &numerator, &denominator)))
      continue;
    if (width == args.width && height == args.height && numerator == args.fps && denominator == 1) {
      Check(reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr,
                                        type.Get()), "SetCurrentMediaType");
      return;
    }
  }
  throw std::runtime_error("exact native MJPG mode is unavailable");
}

uint64_t QpcNanoseconds() {
  LARGE_INTEGER counter{}, frequency{};
  QueryPerformanceCounter(&counter);
  QueryPerformanceFrequency(&frequency);
  return static_cast<uint64_t>((static_cast<long double>(counter.QuadPart) * 1000000000.0L) /
                               static_cast<long double>(frequency.QuadPart));
}

void WriteReady(const Args& args, const std::wstring& link) {
  const std::string json = "{\"controls_locked\":true,\"device_locator\":\"" +
      JsonEscape(Utf8(args.locator)) + "\",\"durable_fingerprint\":\"" +
      args.fingerprint + "\",\"fps_denominator\":1,\"fps_numerator\":" +
      std::to_string(args.fps) + ",\"height\":" + std::to_string(args.height) +
      ",\"kind\":\"mainline1_native_capture_ready\",\"native_handle_ref\":\"" +
      JsonEscape(Utf8(link)) + "\",\"pixel_format\":\"MJPG\",\"schema_version\":\"1.0\",\"width\":" +
      std::to_string(args.width) + "}";
  std::vector<uint8_t> prefix;
  PutBe32(prefix, static_cast<uint32_t>(json.size()));
  WriteAll(prefix.data(), prefix.size());
  WriteAll(reinterpret_cast<const uint8_t*>(json.data()), json.size());
}

void WriteFrame(uint64_t frame_id, uint64_t timestamp_ns, uint32_t width,
                uint32_t height, const uint8_t* payload, size_t payload_size) {
  if (payload_size == 0 || payload_size > UINT32_MAX)
    throw std::runtime_error("invalid MJPEG sample size");
  const auto digest = Sha256(payload, payload_size);
  std::vector<uint8_t> header{'B','A','U','V','C','0','0','1'};
  PutBe64(header, frame_id);
  PutBe64(header, timestamp_ns);
  PutBe32(header, width);
  PutBe32(header, height);
  PutBe32(header, static_cast<uint32_t>(payload_size));
  header.insert(header.end(), digest.begin(), digest.end());
  WriteAll(header.data(), header.size());
  WriteAll(payload, payload_size);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    _setmode(_fileno(stdout), _O_BINARY);
    const Args args = ParseArgs(argc, argv);
    Check(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "CoInitializeEx");
    Check(MFStartup(MF_VERSION), "MFStartup");
    const SelectedSource selected = SelectSource(args.locator);
    ComPtr<IMFMediaSource> source;
    Check(selected.activation->ActivateObject(IID_PPV_ARGS(&source)), "ActivateObject");
    if (!LockControls(source.Get(), args.controls))
      throw std::runtime_error("UVC controls were not explicitly locked and read back");
    ComPtr<IMFSourceReader> reader;
    Check(MFCreateSourceReaderFromMediaSource(source.Get(), nullptr, &reader),
          "MFCreateSourceReaderFromMediaSource");
    SelectNativeMode(reader.Get(), args);
    std::atomic_bool stop{false};
    WriteReady(args, selected.symbolic_link);
    std::thread cancellation([&] {
      HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
      while (!stop) {
        DWORD available = 0;
        if (PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr) &&
            available > 0) {
          char command[16]{};
          DWORD read = 0;
          if (ReadFile(input, command, sizeof(command), &read, nullptr) && read > 0 &&
              std::string(command, read).find("STOP") != std::string::npos) {
            stop = true;
            reader->Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
            return;
          }
        }
        Sleep(10);
      }
    });
    uint64_t frame_id = 0;
    try {
      while (!stop) {
        DWORD flags = 0;
        ComPtr<IMFSample> sample;
        const HRESULT status = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
                                                  nullptr, &flags, nullptr, &sample);
        if (stop) break;
        Check(status, "ReadSample");
        if ((flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM)) != 0 ||
            !sample) throw std::runtime_error("capture source stopped");
        ComPtr<IMFMediaBuffer> buffer;
        Check(sample->ConvertToContiguousBuffer(&buffer), "ConvertToContiguousBuffer");
        BYTE* bytes = nullptr;
        DWORD length = 0;
        Check(buffer->Lock(&bytes, nullptr, &length), "IMFMediaBuffer::Lock");
        try {
          WriteFrame(frame_id++, QpcNanoseconds(), args.width, args.height, bytes, length);
        } catch (...) {
          buffer->Unlock();
          throw;
        }
        buffer->Unlock();
      }
    } catch (...) {
      stop = true;
      cancellation.join();
      throw;
    }
    stop = true;
    cancellation.join();
    source->Shutdown();
    MFShutdown();
    CoUninitialize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "{\"kind\":\"native_capture_error\",\"message\":\""
              << JsonEscape(error.what()) << "\"}" << std::endl;
    return 2;
  }
}
