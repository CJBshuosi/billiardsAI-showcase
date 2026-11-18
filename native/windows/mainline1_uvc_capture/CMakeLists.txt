cmake_minimum_required(VERSION 3.24)
project(mainline1_uvc_capture LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "mainline1_uvc_capture is Windows-only")
endif()

add_executable(mainline1_uvc_capture main.cpp)
target_compile_features(mainline1_uvc_capture PRIVATE cxx_std_20)
target_compile_definitions(mainline1_uvc_capture PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(mainline1_uvc_capture PRIVATE /W4 /WX /permissive-)
target_link_libraries(mainline1_uvc_capture PRIVATE
  mfplat mfreadwrite mfuuid ole32 setupapi bcrypt strmiids
)
