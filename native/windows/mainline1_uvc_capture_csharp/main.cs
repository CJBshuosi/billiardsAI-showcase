using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

internal static class NativeCaptureProgram
{
    private const int S_OK = 0;
    private const int S_FALSE = 1;
    private const int DigcfPresent = 0x2;
    private const int DigcfDeviceInterface = 0x10;
    private const int ErrorInsufficientBuffer = 122;
    private const int CameraAuto = 1;
    private const int CameraManual = 2;
    private const int ProcAuto = 1;
    private const int ProcManual = 2;
    private const int FourCcMjpg = 0x47504A4D;

    private static readonly Guid ClsidSystemDeviceEnum = new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86");
    private static readonly Guid ClsidFilterGraph = new Guid("E436EBB3-524F-11CE-9F53-0020AF0BA770");
    private static readonly Guid ClsidCaptureGraphBuilder2 = new Guid("BF87B6E1-8C27-11D0-B3F0-00AA003761C5");
    private static readonly Guid ClsidSampleGrabber = new Guid("C1F400A0-3F08-11D3-9F0B-006008039E37");
    private static readonly Guid ClsidNullRenderer = new Guid("C1F400A4-3F08-11D3-9F0B-006008039E37");
    private static readonly Guid VideoInputDeviceCategory = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
    private static readonly Guid PinCategoryCapture = new Guid("FB6C4281-0353-11D1-905F-0000C0CC16BA");
    private static readonly Guid MediaTypeVideo = new Guid("73646976-0000-0010-8000-00AA00389B71");
    private static readonly Guid MediaSubtypeMjpg = new Guid("47504A4D-0000-0010-8000-00AA00389B71");
    private static readonly Guid FormatVideoInfo = new Guid("05589F80-C356-11CE-BF01-00AA0055595A");

    public static int Main(string[] argv)
    {
        StreamWriter errorWriter = new StreamWriter(Console.OpenStandardError(), new UTF8Encoding(false));
        errorWriter.AutoFlush = true;
        Console.SetError(errorWriter);
        try
        {
            CaptureArguments args = CaptureArguments.Parse(argv);
            using (CaptureGraph graph = CaptureGraph.Open(args))
            {
                FramePacket firstFrame = graph.TakeFrame(5000);
                if (firstFrame == null)
                    throw new InvalidOperationException("native camera did not deliver its first frame within 5000 ms");
                WriteReady(args, graph.DevicePath, graph.DurableFingerprint);
                Thread inputThread = new Thread(delegate()
                {
                    try
                    {
                        string line;
                        while ((line = Console.In.ReadLine()) != null)
                        {
                            if (line.Trim().Equals("STOP", StringComparison.Ordinal))
                            {
                                graph.RequestStop();
                                return;
                            }
                        }
                        graph.RequestStop();
                    }
                    catch
                    {
                        graph.RequestStop();
                    }
                });
                inputThread.IsBackground = true;
                inputThread.Name = "mainline1-native-stop";
                inputThread.Start();

                ulong frameId = 0;
                WriteFrame(frameId++, firstFrame.TimestampNs, args.Width, args.Height, firstFrame.Payload);
                while (!graph.StopRequested)
                {
                    FramePacket frame = graph.TakeFrame(500);
                    if (frame == null)
                    {
                        if (graph.StopRequested) break;
                        throw new InvalidOperationException("native camera read exceeded 500 ms");
                    }
                    WriteFrame(frameId++, frame.TimestampNs, args.Width, args.Height, frame.Payload);
                }
                if (graph.FailureMessage != null)
                    throw new InvalidOperationException(graph.FailureMessage);
            }
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("{\"exception_type\":\"" + JsonEscape(error.GetType().FullName) +
                "\",\"hresult\":" + error.HResult.ToString(System.Globalization.CultureInfo.InvariantCulture) +
                ",\"kind\":\"native_capture_error\",\"message\":\"" + JsonEscape(error.Message) + "\"}");
            return 2;
        }
    }

    private static void WriteReady(CaptureArguments args, string devicePath, string durableFingerprint)
    {
        string json = "{\"controls_locked\":true,\"device_locator\":\"" +
            JsonEscape(args.DeviceLocator) + "\",\"durable_fingerprint\":\"" +
            durableFingerprint + "\",\"fps_denominator\":1,\"fps_numerator\":" +
            args.Fps.ToString(System.Globalization.CultureInfo.InvariantCulture) +
            ",\"height\":" + args.Height.ToString(System.Globalization.CultureInfo.InvariantCulture) +
            ",\"kind\":\"mainline1_native_capture_ready\",\"native_handle_ref\":\"dshow://" +
            JsonEscape(devicePath) + "\",\"pixel_format\":\"MJPG\",\"schema_version\":\"1.0\",\"width\":" +
            args.Width.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
        byte[] encoded = Encoding.UTF8.GetBytes(json);
        Stream output = Console.OpenStandardOutput();
        WriteBe32(output, (uint)encoded.Length);
        output.Write(encoded, 0, encoded.Length);
        output.Flush();
    }

    private static void WriteFrame(ulong frameId, ulong timestampNs, int width, int height, byte[] payload)
    {
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(payload);
        Stream output = Console.OpenStandardOutput();
        byte[] magic = Encoding.ASCII.GetBytes("BAUVC001");
        output.Write(magic, 0, magic.Length);
        WriteBe64(output, frameId);
        WriteBe64(output, timestampNs);
        WriteBe32(output, (uint)width);
        WriteBe32(output, (uint)height);
        WriteBe32(output, (uint)payload.Length);
        output.Write(digest, 0, digest.Length);
        output.Write(payload, 0, payload.Length);
        output.Flush();
    }

    private static void WriteBe32(Stream stream, uint value)
    {
        stream.WriteByte((byte)(value >> 24));
        stream.WriteByte((byte)(value >> 16));
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)value);
    }

    private static void WriteBe64(Stream stream, ulong value)
    {
        for (int shift = 56; shift >= 0; shift -= 8) stream.WriteByte((byte)(value >> shift));
    }

    private static string JsonEscape(string value)
    {
        StringBuilder output = new StringBuilder();
        foreach (char c in value)
        {
            if (c == '\\' || c == '"') output.Append('\\').Append(c);
            else if (c >= 0x20) output.Append(c);
        }
        return output.ToString();
    }

    private static string CanonicalIdentityValue(string value)
    {
        if (value == null) return "null";
        string normalized = value.Trim().ToLowerInvariant();
        // Windows PnP identifiers are ASCII. Reject unsupported identities
        // rather than diverge from Python casefold / ensure_ascii semantics.
        foreach (char c in normalized)
            if (c < 0x20 || c > 0x7e)
                throw new InvalidOperationException("device identity contains unsupported characters");
        return "\"" + JsonEscape(normalized) + "\"";
    }

    private static string DeviceFingerprint(string locator)
    {
        // Keep the same VID/PID/optional-serial extraction and sorted JSON
        // fields as enumerate_usb_devices / UsbDeviceIdentity in Python.
        Match match = Regex.Match(locator, @"VID_([0-9A-F]{4})&PID_([0-9A-F]{4})(?:\\([^\\]+))?",
            RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        string vendor = match.Success ? match.Groups[1].Value : null;
        string product = match.Success ? match.Groups[2].Value : null;
        string serial = match.Success && match.Groups[3].Success ? match.Groups[3].Value : null;
        string canonical = "{\"device_locator\":" + CanonicalIdentityValue(locator) + "," +
            "\"product_id\":" + CanonicalIdentityValue(product) + "," +
            "\"serial_number\":" + CanonicalIdentityValue(serial) + "," +
            "\"vendor_id\":" + CanonicalIdentityValue(vendor) + "}";
        byte[] digest;
        using (SHA256 sha = SHA256.Create()) digest = sha.ComputeHash(Encoding.UTF8.GetBytes(canonical));
        return BitConverter.ToString(digest).Replace("-", "").ToLowerInvariant();
    }

    internal static void Check(int hr, string operation)
    {
        if (hr < 0) Marshal.ThrowExceptionForHR(hr, new IntPtr(-1));
    }

    private sealed class CaptureArguments
    {
        public string DeviceLocator;
        public string Fingerprint;
        public int Width = 1920;
        public int Height = 1080;
        public int Fps = 60;
        public readonly List<ControlRequest> Controls = new List<ControlRequest>();

        public static CaptureArguments Parse(string[] argv)
        {
            CaptureArguments args = new CaptureArguments();
            for (int index = 0; index < argv.Length; index += 2)
            {
                if (index + 1 >= argv.Length) throw new ArgumentException("missing argument value");
                string name = argv[index];
                string value = argv[index + 1];
                if (name == "--device-locator") args.DeviceLocator = value;
                else if (name == "--fingerprint") args.Fingerprint = value;
                else if (name == "--width") args.Width = Int32.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
                else if (name == "--height") args.Height = Int32.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
                else if (name == "--fps") args.Fps = Int32.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
                else if (name == "--pixel-format" && value != "MJPG") throw new ArgumentException("only MJPG is supported");
                else if (name == "--control") args.Controls.Add(ControlRequest.Parse(value));
                else if (name != "--pixel-format") throw new ArgumentException("unknown argument: " + name);
            }
            if (String.IsNullOrWhiteSpace(args.DeviceLocator)) throw new ArgumentException("device locator is required");
            if (String.IsNullOrEmpty(args.Fingerprint) || !Regex.IsMatch(args.Fingerprint, @"\A[0-9a-f]{64}\z"))
                throw new ArgumentException("lowercase SHA-256 fingerprint is required");
            if (args.Width <= 0 || args.Height <= 0 || args.Fps <= 0) throw new ArgumentException("capture geometry and FPS must be positive");
            if (args.Controls.Count == 0) throw new ArgumentException("explicit controls are required");
            return args;
        }
    }

    private sealed class ControlRequest
    {
        public string Name;
        public int Value;

        public static ControlRequest Parse(string text)
        {
            int equal = text.IndexOf('=');
            if (equal <= 0 || equal == text.Length - 1) throw new ArgumentException("invalid control: " + text);
            return new ControlRequest
            {
                Name = text.Substring(0, equal),
                Value = Int32.Parse(text.Substring(equal + 1), System.Globalization.CultureInfo.InvariantCulture)
            };
        }
    }

    private sealed class FramePacket
    {
        public ulong TimestampNs;
        public byte[] Payload;
    }

    [ComVisible(true), ClassInterface(ClassInterfaceType.None)]
    private sealed class CaptureGraph : IDisposable, ISampleGrabberCB
    {
        private readonly object queueLock = new object();
        private readonly Queue<FramePacket> frames = new Queue<FramePacket>();
        private object graphObject;
        private object builderObject;
        private object sourceObject;
        private object grabberObject;
        private object nullObject;
        private IMediaControl mediaControl;
        private volatile bool stopRequested;
        private string failureMessage;
        private bool disposed;

        public string DevicePath { get; private set; }
        public string DurableFingerprint { get; private set; }
        public bool StopRequested { get { return stopRequested; } }
        public string FailureMessage { get { return failureMessage; } }

        public static CaptureGraph Open(CaptureArguments args)
        {
            CaptureGraph graph = new CaptureGraph();
            try
            {
                graph.Initialize(args);
                return graph;
            }
            catch
            {
                graph.Dispose();
                throw;
            }
        }

        private void Initialize(CaptureArguments args)
        {
            DeviceSelection selected = SelectDevice(args.DeviceLocator, args.Fingerprint);
            DevicePath = selected.DevicePath;
            DurableFingerprint = selected.DurableFingerprint;
            sourceObject = selected.Filter;

            graphObject = Activator.CreateInstance(Type.GetTypeFromCLSID(ClsidFilterGraph, true));
            builderObject = Activator.CreateInstance(Type.GetTypeFromCLSID(ClsidCaptureGraphBuilder2, true));
            grabberObject = Activator.CreateInstance(Type.GetTypeFromCLSID(ClsidSampleGrabber, true));
            nullObject = Activator.CreateInstance(Type.GetTypeFromCLSID(ClsidNullRenderer, true));

            IGraphBuilder graph = (IGraphBuilder)graphObject;
            ICaptureGraphBuilder2 builder = (ICaptureGraphBuilder2)builderObject;
            Check(builder.SetFiltergraph(graph), "SetFiltergraph");
            Check(graph.AddFilter((IBaseFilter)sourceObject, "TrustedCaptureSource"), "Add source filter");
            Check(graph.AddFilter((IBaseFilter)grabberObject, "MjpegSampleGrabber"), "Add sample grabber");
            Check(graph.AddFilter((IBaseFilter)nullObject, "NullRenderer"), "Add null renderer");

            ConfigureExactMode(builder, (IBaseFilter)sourceObject, args);
            LockControls(sourceObject, args.Controls);

            ISampleGrabber grabber = (ISampleGrabber)grabberObject;
            AMMediaType requested = new AMMediaType();
            requested.majorType = MediaTypeVideo;
            requested.subType = MediaSubtypeMjpg;
            requested.formatType = Guid.Empty;
            Check(grabber.SetMediaType(requested), "SampleGrabber SetMediaType");
            Check(grabber.SetOneShot(false), "SampleGrabber SetOneShot");
            Check(grabber.SetBufferSamples(false), "SampleGrabber SetBufferSamples");
            Check(grabber.SetCallback(this, 1), "SampleGrabber SetCallback");

            Guid pin = PinCategoryCapture;
            Guid media = MediaTypeVideo;
            Check(builder.RenderStream(ref pin, ref media, sourceObject, (IBaseFilter)grabberObject, (IBaseFilter)nullObject), "RenderStream");
            mediaControl = (IMediaControl)graphObject;
            Check(mediaControl.Run(), "IMediaControl Run");
        }

        public FramePacket TakeFrame(int timeoutMilliseconds)
        {
            lock (queueLock)
            {
                if (frames.Count == 0 && !stopRequested) Monitor.Wait(queueLock, timeoutMilliseconds);
                if (frames.Count == 0) return null;
                return frames.Dequeue();
            }
        }

        public void RequestStop()
        {
            stopRequested = true;
            lock (queueLock) Monitor.PulseAll(queueLock);
        }

        public int SampleCB(double sampleTime, IMediaSample sample) { return S_OK; }

        public int BufferCB(double sampleTime, IntPtr buffer, int bufferLength)
        {
            try
            {
                if (stopRequested) return S_OK;
                if (buffer == IntPtr.Zero || bufferLength <= 0) throw new InvalidOperationException("empty MJPG sample");
                byte[] payload = new byte[bufferLength];
                Marshal.Copy(buffer, payload, 0, bufferLength);
                long ticks = Stopwatch.GetTimestamp();
                ulong timestamp = (ulong)((double)ticks * 1000000000.0 / Stopwatch.Frequency);
                lock (queueLock)
                {
                    if (frames.Count >= 32)
                    {
                        failureMessage = "native capture callback queue exceeded 32 frames";
                        stopRequested = true;
                        Monitor.PulseAll(queueLock);
                        return unchecked((int)0x80004005);
                    }
                    frames.Enqueue(new FramePacket { TimestampNs = timestamp, Payload = payload });
                    Monitor.Pulse(queueLock);
                }
                return S_OK;
            }
            catch (Exception error)
            {
                failureMessage = "native capture callback failed: " + error.Message;
                stopRequested = true;
                lock (queueLock) Monitor.PulseAll(queueLock);
                return unchecked((int)0x80004005);
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            stopRequested = true;
            try { if (mediaControl != null) mediaControl.Stop(); } catch { }
            ReleaseCom(nullObject);
            ReleaseCom(grabberObject);
            ReleaseCom(sourceObject);
            ReleaseCom(builderObject);
            ReleaseCom(graphObject);
        }

        private static void ReleaseCom(object value)
        {
            if (value != null && Marshal.IsComObject(value))
            {
                try { Marshal.ReleaseComObject(value); } catch { }
            }
        }

        private static DeviceSelection SelectDevice(string expectedLocator, string expectedFingerprint)
        {
            object enumeratorObject = Activator.CreateInstance(Type.GetTypeFromCLSID(ClsidSystemDeviceEnum, true));
            ICreateDevEnum devices = (ICreateDevEnum)enumeratorObject;
            IEnumMoniker monikers = null;
            Guid category = VideoInputDeviceCategory;
            int result = devices.CreateClassEnumerator(ref category, out monikers, 0);
            if (result == S_FALSE || monikers == null) throw new InvalidOperationException("no DirectShow video devices were found");
            Check(result, "CreateClassEnumerator");
            DeviceSelection selected = null;
            IMoniker[] current = new IMoniker[1];
            IntPtr fetched = Marshal.AllocCoTaskMem(sizeof(int));
            try
            {
                while (monikers.Next(1, current, fetched) == S_OK)
                {
                    IMoniker moniker = current[0];
                    try
                    {
                        string path = ReadDevicePath(moniker);
                        if (String.IsNullOrWhiteSpace(path)) continue;
                        string locator = ResolvePnpInstance(path);
                        if (!locator.Equals(expectedLocator, StringComparison.OrdinalIgnoreCase)) continue;
                        string fingerprint = DeviceFingerprint(locator);
                        if (!fingerprint.Equals(expectedFingerprint, StringComparison.Ordinal))
                            throw new InvalidOperationException("resolved camera fingerprint does not match capture setup");
                        if (selected != null) throw new InvalidOperationException("PnP locator mapped to more than one capture handle");
                        Guid filterGuid = typeof(IBaseFilter).GUID;
                        object filter;
                        moniker.BindToObject(null, null, ref filterGuid, out filter);
                        selected = new DeviceSelection
                        {
                            DevicePath = path, DurableFingerprint = fingerprint, Filter = filter
                        };
                    }
                    finally
                    {
                        if (moniker != null && Marshal.IsComObject(moniker)) Marshal.ReleaseComObject(moniker);
                        current[0] = null;
                    }
                }
            }
            finally
            {
                Marshal.FreeCoTaskMem(fetched);
                ReleaseCom(monikers);
                ReleaseCom(enumeratorObject);
            }
            if (selected == null) throw new InvalidOperationException("PnP locator has no exact DirectShow capture handle");
            return selected;
        }

        private static string ReadDevicePath(IMoniker moniker)
        {
            Guid bagGuid = typeof(IPropertyBag).GUID;
            object bagObject;
            moniker.BindToStorage(null, null, ref bagGuid, out bagObject);
            try
            {
                object value;
                int hr = ((IPropertyBag)bagObject).Read("DevicePath", out value, IntPtr.Zero);
                if (hr < 0 || value == null) return null;
                return Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture);
            }
            finally { ReleaseCom(bagObject); }
        }

        private static string ResolvePnpInstance(string devicePath)
        {
            int openBrace = devicePath.LastIndexOf('{');
            int closeBrace = devicePath.IndexOf('}', openBrace + 1);
            Guid interfaceClass;
            if (openBrace < 0 || closeBrace < 0 || !Guid.TryParse(devicePath.Substring(openBrace, closeBrace - openBrace + 1), out interfaceClass))
                throw new InvalidOperationException("capture handle has no device-interface class GUID");
            IntPtr set = SetupDiGetClassDevs(ref interfaceClass, null, IntPtr.Zero, DigcfPresent | DigcfDeviceInterface);
            if (set == new IntPtr(-1)) throw new InvalidOperationException("SetupDiGetClassDevs failed");
            try
            {
                SP_DEVICE_INTERFACE_DATA interfaceData = new SP_DEVICE_INTERFACE_DATA();
                interfaceData.cbSize = Marshal.SizeOf(typeof(SP_DEVICE_INTERFACE_DATA));
                if (!SetupDiOpenDeviceInterface(set, devicePath, 0, ref interfaceData))
                    throw new InvalidOperationException("capture DevicePath was rejected by SetupAPI: " + Marshal.GetLastWin32Error());
                SP_DEVINFO_DATA deviceData = new SP_DEVINFO_DATA();
                deviceData.cbSize = Marshal.SizeOf(typeof(SP_DEVINFO_DATA));
                int required;
                SetupDiGetDeviceInterfaceDetail(set, ref interfaceData, IntPtr.Zero, 0, out required, ref deviceData);
                if (Marshal.GetLastWin32Error() != ErrorInsufficientBuffer || required <= 0)
                    throw new InvalidOperationException("SetupDiGetDeviceInterfaceDetail sizing failed");
                IntPtr detail = Marshal.AllocHGlobal(required);
                try
                {
                    Marshal.WriteInt32(detail, IntPtr.Size == 8 ? 8 : 6);
                    if (!SetupDiGetDeviceInterfaceDetail(set, ref interfaceData, detail, required, out required, ref deviceData))
                        throw new InvalidOperationException("SetupDiGetDeviceInterfaceDetail failed: " + Marshal.GetLastWin32Error());
                    StringBuilder instance = new StringBuilder(512);
                    if (!SetupDiGetDeviceInstanceId(set, ref deviceData, instance, instance.Capacity, IntPtr.Zero))
                        throw new InvalidOperationException("SetupDiGetDeviceInstanceId failed: " + Marshal.GetLastWin32Error());
                    return instance.ToString();
                }
                finally { Marshal.FreeHGlobal(detail); }
            }
            finally { SetupDiDestroyDeviceInfoList(set); }
        }

        private static void ConfigureExactMode(ICaptureGraphBuilder2 builder, IBaseFilter source, CaptureArguments args)
        {
            Guid pin = PinCategoryCapture;
            Guid media = MediaTypeVideo;
            Guid configGuid = typeof(IAMStreamConfig).GUID;
            object configObject;
            Check(builder.FindInterface(pin, media, source, configGuid, out configObject), "FindInterface IAMStreamConfig");
            try
            {
                IAMStreamConfig config = (IAMStreamConfig)configObject;
                int count;
                int capabilityBytes;
                Check(config.GetNumberOfCapabilities(out count, out capabilityBytes), "GetNumberOfCapabilities");
                if (count <= 0 || capabilityBytes <= 0 || capabilityBytes > 4096) throw new InvalidOperationException("invalid stream capability inventory");
                IntPtr capabilities = Marshal.AllocCoTaskMem(capabilityBytes);
                try
                {
                    for (int index = 0; index < count; ++index)
                    {
                        IntPtr mediaPointer;
                        int hr = config.GetStreamCaps(index, out mediaPointer, capabilities);
                        if (hr < 0 || mediaPointer == IntPtr.Zero) continue;
                        AMMediaType candidate = (AMMediaType)Marshal.PtrToStructure(mediaPointer, typeof(AMMediaType));
                        try
                        {
                            if (candidate.majorType != MediaTypeVideo || candidate.subType != MediaSubtypeMjpg || candidate.formatType != FormatVideoInfo || candidate.formatPtr == IntPtr.Zero || candidate.formatSize < Marshal.SizeOf(typeof(VideoInfoHeader))) continue;
                            VideoInfoHeader header = (VideoInfoHeader)Marshal.PtrToStructure(candidate.formatPtr, typeof(VideoInfoHeader));
                            double actualFps = header.AvgTimePerFrame > 0 ? 10000000.0 / header.AvgTimePerFrame : 0.0;
                            if (header.BmiHeader.Width != args.Width || Math.Abs(header.BmiHeader.Height) != args.Height || header.BmiHeader.Compression != FourCcMjpg || Math.Abs(actualFps - args.Fps) > 0.1) continue;
                            Check(config.SetFormat(candidate), "SetFormat exact MJPG mode");
                            return;
                        }
                        finally { FreeMediaType(mediaPointer, candidate); }
                    }
                }
                finally { Marshal.FreeCoTaskMem(capabilities); }
                throw new InvalidOperationException("exact native 1920x1080 MJPG 60 FPS mode is unavailable");
            }
            finally { GC.KeepAlive(configObject); }
        }

        private static void FreeMediaType(IntPtr pointer, AMMediaType media)
        {
            if (media.formatPtr != IntPtr.Zero) Marshal.FreeCoTaskMem(media.formatPtr);
            if (media.unkPtr != IntPtr.Zero) Marshal.Release(media.unkPtr);
            Marshal.FreeCoTaskMem(pointer);
        }

        private static void LockControls(object source, IList<ControlRequest> controls)
        {
            IAMCameraControl camera = source as IAMCameraControl;
            IAMVideoProcAmp proc = source as IAMVideoProcAmp;
            foreach (ControlRequest request in controls)
            {
                int property;
                int actual;
                int flags;
                if (request.Name == "auto_focus" || request.Name == "auto_exposure")
                {
                    if (camera == null) throw new InvalidOperationException("camera control interface is unavailable");
                    property = request.Name == "auto_focus" ? 6 : 4;
                    Check(camera.Get(property, out actual, out flags), "CameraControl Get auto mode");
                    int requestedFlags = request.Value == 0 ? CameraManual : CameraAuto;
                    Check(camera.Set(property, actual, requestedFlags), "CameraControl Set auto mode");
                    Check(camera.Get(property, out actual, out flags), "CameraControl readback auto mode");
                    if (flags != requestedFlags) throw new InvalidOperationException(request.Name + " did not read back exactly");
                    continue;
                }
                if (request.Name == "auto_white_balance")
                {
                    if (proc == null) throw new InvalidOperationException("video processing control interface is unavailable");
                    property = 7;
                    Check(proc.Get(property, out actual, out flags), "VideoProcAmp Get auto white balance");
                    int requestedFlags = request.Value == 0 ? ProcManual : ProcAuto;
                    Check(proc.Set(property, actual, requestedFlags), "VideoProcAmp Set auto white balance");
                    Check(proc.Get(property, out actual, out flags), "VideoProcAmp readback auto white balance");
                    if (flags != requestedFlags) throw new InvalidOperationException("auto_white_balance did not read back exactly");
                    continue;
                }
                if (TryCameraProperty(request.Name, out property))
                {
                    if (camera == null) throw new InvalidOperationException("camera control interface is unavailable");
                    Check(camera.Set(property, request.Value, CameraManual), "CameraControl Set " + request.Name);
                    Check(camera.Get(property, out actual, out flags), "CameraControl readback " + request.Name);
                    if (actual != request.Value || flags != CameraManual) throw new InvalidOperationException(request.Name + " did not read back exactly");
                }
                else if (TryProcProperty(request.Name, out property))
                {
                    if (proc == null) throw new InvalidOperationException("video processing control interface is unavailable");
                    Check(proc.Set(property, request.Value, ProcManual), "VideoProcAmp Set " + request.Name);
                    Check(proc.Get(property, out actual, out flags), "VideoProcAmp readback " + request.Name);
                    if (actual != request.Value || flags != ProcManual) throw new InvalidOperationException(request.Name + " did not read back exactly");
                }
                else throw new InvalidOperationException("unsupported control: " + request.Name);
            }
        }

        private static bool TryCameraProperty(string name, out int property)
        {
            if (name == "exposure") property = 4;
            else if (name == "focus") property = 6;
            else if (name == "zoom") property = 3;
            else { property = -1; return false; }
            return true;
        }

        private static bool TryProcProperty(string name, out int property)
        {
            if (name == "white_balance_temperature") property = 7;
            else if (name == "gain") property = 9;
            else if (name == "brightness") property = 0;
            else if (name == "contrast") property = 1;
            else if (name == "saturation") property = 3;
            else if (name == "sharpness") property = 4;
            else { property = -1; return false; }
            return true;
        }
    }

    private sealed class DeviceSelection
    {
        public string DurableFingerprint;
        public string DevicePath;
        public object Filter;
    }

    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr SetupDiGetClassDevs(ref Guid classGuid, string enumerator, IntPtr hwndParent, int flags);
    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetupDiOpenDeviceInterface(IntPtr deviceInfoSet, string devicePath, int openFlags, ref SP_DEVICE_INTERFACE_DATA deviceInterfaceData);
    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetupDiGetDeviceInterfaceDetail(IntPtr deviceInfoSet, ref SP_DEVICE_INTERFACE_DATA deviceInterfaceData, IntPtr detail, int detailBytes, out int requiredBytes, ref SP_DEVINFO_DATA deviceInfoData);
    [DllImport("setupapi.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetupDiGetDeviceInstanceId(IntPtr deviceInfoSet, ref SP_DEVINFO_DATA deviceInfoData, StringBuilder deviceInstanceId, int deviceInstanceIdBytes, IntPtr requiredBytes);
    [DllImport("setupapi.dll", SetLastError = true)]
    private static extern bool SetupDiDestroyDeviceInfoList(IntPtr deviceInfoSet);

    [StructLayout(LayoutKind.Sequential)]
    private struct SP_DEVICE_INTERFACE_DATA { public int cbSize; public Guid InterfaceClassGuid; public int Flags; public UIntPtr Reserved; }
    [StructLayout(LayoutKind.Sequential)]
    private struct SP_DEVINFO_DATA { public int cbSize; public Guid ClassGuid; public int DevInst; public UIntPtr Reserved; }

    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ICreateDevEnum
    {
        [PreserveSig] int CreateClassEnumerator([In] ref Guid type, out IEnumMoniker enumerator, int flags);
    }

    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IPropertyBag
    {
        [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name, [MarshalAs(UnmanagedType.Struct)] out object value, IntPtr errorLog);
        [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name, [MarshalAs(UnmanagedType.Struct)] ref object value);
    }

    [ComImport, Guid("56A86895-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IBaseFilter
    {
        [PreserveSig] int GetClassID(out Guid classId);
        [PreserveSig] int Stop();
        [PreserveSig] int Pause();
        [PreserveSig] int Run(long start);
        [PreserveSig] int GetState(int milliseconds, out int state);
        [PreserveSig] int SetSyncSource(IntPtr clock);
        [PreserveSig] int GetSyncSource(out IntPtr clock);
        [PreserveSig] int EnumPins(out IntPtr pins);
        [PreserveSig] int FindPin([MarshalAs(UnmanagedType.LPWStr)] string id, out IntPtr pin);
        [PreserveSig] int QueryFilterInfo(out FilterInfo info);
        [PreserveSig] int JoinFilterGraph(IntPtr graph, [MarshalAs(UnmanagedType.LPWStr)] string name);
        [PreserveSig] int QueryVendorInfo(out IntPtr vendorInfo);
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct FilterInfo { [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Name; public IntPtr Graph; }

    [ComImport, Guid("56A868A9-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IGraphBuilder
    {
        [PreserveSig] int AddFilter([In] IBaseFilter filter, [MarshalAs(UnmanagedType.LPWStr)] string name);
        [PreserveSig] int RemoveFilter([In] IBaseFilter filter);
        [PreserveSig] int EnumFilters(out IntPtr filters);
        [PreserveSig] int FindFilterByName([MarshalAs(UnmanagedType.LPWStr)] string name, out IBaseFilter filter);
        [PreserveSig] int ConnectDirect(IntPtr outputPin, IntPtr inputPin, IntPtr mediaType);
        [PreserveSig] int Reconnect(IntPtr pin);
        [PreserveSig] int Disconnect(IntPtr pin);
        [PreserveSig] int SetDefaultSyncSource();
        [PreserveSig] int Connect(IntPtr outputPin, IntPtr inputPin);
        [PreserveSig] int Render(IntPtr outputPin);
        [PreserveSig] int RenderFile([MarshalAs(UnmanagedType.LPWStr)] string file, [MarshalAs(UnmanagedType.LPWStr)] string playList);
        [PreserveSig] int AddSourceFilter([MarshalAs(UnmanagedType.LPWStr)] string fileName, [MarshalAs(UnmanagedType.LPWStr)] string filterName, out IBaseFilter filter);
        [PreserveSig] int SetLogFile(IntPtr fileHandle);
        [PreserveSig] int Abort();
        [PreserveSig] int ShouldOperationContinue();
    }

    [ComImport, Guid("93E5A4E0-2D50-11D2-ABFA-00A0C9C6E38D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ICaptureGraphBuilder2
    {
        [PreserveSig] int SetFiltergraph([In] IGraphBuilder graph);
        [PreserveSig] int GetFiltergraph(out IGraphBuilder graph);
        [PreserveSig] int SetOutputFileName([In] ref Guid type, [MarshalAs(UnmanagedType.LPWStr)] string fileName, out IBaseFilter filter, out IntPtr sink);
        [PreserveSig] int FindInterface(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid category,
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid type,
            [In] IBaseFilter source,
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid interfaceId,
            [Out, MarshalAs(UnmanagedType.IUnknown)] out object found);
        [PreserveSig] int RenderStream([In] ref Guid category, [In] ref Guid type, [In, MarshalAs(UnmanagedType.IUnknown)] object source, [In] IBaseFilter compressor, [In] IBaseFilter renderer);
        [PreserveSig] int ControlStream([In] ref Guid category, [In] ref Guid type, [In] IBaseFilter filter, long start, long stop, short startCookie, short stopCookie);
        [PreserveSig] int AllocCapFile([MarshalAs(UnmanagedType.LPWStr)] string fileName, long size);
        [PreserveSig] int CopyCaptureFile([MarshalAs(UnmanagedType.LPWStr)] string oldFile, [MarshalAs(UnmanagedType.LPWStr)] string newFile, [MarshalAs(UnmanagedType.Bool)] bool allowEscAbort, IntPtr callback);
        [PreserveSig] int FindPin([In, MarshalAs(UnmanagedType.IUnknown)] object source, int direction, [In] ref Guid category, [In] ref Guid type, [MarshalAs(UnmanagedType.Bool)] bool unconnected, int index, out IntPtr pin);
    }

    [ComImport, Guid("56A868B1-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsDual)]
    private interface IMediaControl
    {
        [PreserveSig] int Run();
        [PreserveSig] int Pause();
        [PreserveSig] int Stop();
        [PreserveSig] int GetState(int milliseconds, out int state);
        [PreserveSig] int RenderFile([MarshalAs(UnmanagedType.BStr)] string fileName);
        [PreserveSig] int AddSourceFilter([MarshalAs(UnmanagedType.BStr)] string fileName, out object filterInfo);
        [PreserveSig] int get_FilterCollection(out object collection);
        [PreserveSig] int get_RegFilterCollection(out object collection);
        [PreserveSig] int StopWhenReady();
    }

    [ComImport, Guid("C6E13340-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAMStreamConfig
    {
        [PreserveSig] int SetFormat([In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType mediaType);
        [PreserveSig] int GetFormat(out IntPtr mediaType);
        [PreserveSig] int GetNumberOfCapabilities(out int count, out int size);
        [PreserveSig] int GetStreamCaps(int index, out IntPtr mediaType, IntPtr capabilities);
    }

    [ComImport, Guid("6B652FFF-11FE-4FCE-92AD-0266B5D7C78F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ISampleGrabber
    {
        [PreserveSig] int SetOneShot([MarshalAs(UnmanagedType.Bool)] bool oneShot);
        [PreserveSig] int SetMediaType([In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType mediaType);
        [PreserveSig] int GetConnectedMediaType([Out, MarshalAs(UnmanagedType.LPStruct)] AMMediaType mediaType);
        [PreserveSig] int SetBufferSamples([MarshalAs(UnmanagedType.Bool)] bool bufferThem);
        [PreserveSig] int GetCurrentBuffer(ref int bufferSize, IntPtr buffer);
        [PreserveSig] int GetCurrentSample(out IMediaSample sample);
        [PreserveSig] int SetCallback(ISampleGrabberCB callback, int method);
    }

    [ComVisible(true), Guid("0579154A-2B53-4994-B0D0-E773148EFF85"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface ISampleGrabberCB
    {
        [PreserveSig] int SampleCB(double sampleTime, IMediaSample sample);
        [PreserveSig] int BufferCB(double sampleTime, IntPtr buffer, int bufferLength);
    }

    [ComImport, Guid("56A8689A-0AD4-11CE-B03A-0020AF0BA770"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMediaSample { }

    [ComImport, Guid("C6E13370-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAMCameraControl
    {
        [PreserveSig] int GetRange(int property, out int minimum, out int maximum, out int step, out int defaultValue, out int flags);
        [PreserveSig] int Set(int property, int value, int flags);
        [PreserveSig] int Get(int property, out int value, out int flags);
    }

    [ComImport, Guid("C6E13360-30AC-11D0-A18C-00A0C9118956"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAMVideoProcAmp
    {
        [PreserveSig] int GetRange(int property, out int minimum, out int maximum, out int step, out int defaultValue, out int flags);
        [PreserveSig] int Set(int property, int value, int flags);
        [PreserveSig] int Get(int property, out int value, out int flags);
    }

    [StructLayout(LayoutKind.Sequential)]
    private sealed class AMMediaType
    {
        public Guid majorType;
        public Guid subType;
        [MarshalAs(UnmanagedType.Bool)] public bool fixedSizeSamples;
        [MarshalAs(UnmanagedType.Bool)] public bool temporalCompression;
        public int sampleSize;
        public Guid formatType;
        public IntPtr unkPtr;
        public int formatSize;
        public IntPtr formatPtr;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DsRect { public int Left; public int Top; public int Right; public int Bottom; }
    [StructLayout(LayoutKind.Sequential)]
    private struct BitmapInfoHeader
    {
        public int Size; public int Width; public int Height; public short Planes; public short BitCount;
        public int Compression; public int ImageSize; public int XPelsPerMeter; public int YPelsPerMeter;
        public int ClrUsed; public int ClrImportant;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct VideoInfoHeader
    {
        public DsRect Source; public DsRect Target; public int BitRate; public int BitErrorRate;
        public long AvgTimePerFrame; public BitmapInfoHeader BmiHeader;
    }
}
