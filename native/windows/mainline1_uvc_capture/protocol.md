# Mainline 1 native UVC capture protocol

The helper writes binary data only to stdout and NDJSON diagnostics only to stderr.
All integers on stdout are unsigned big-endian.

The first record is `uint32 json_bytes` followed by one UTF-8 JSON object. The object
must identify schema `1.0`, kind `mainline1_native_capture_ready`, the exact requested
PnP instance ID, the selected Media Foundation symbolic link, the caller-supplied
durable fingerprint, the selected native MJPG width/height/rational FPS, and
`controls_locked: true`. No frame may precede this record.

Every source frame is:

```
8 bytes   magic BAUVC001
uint64    zero-based contiguous frame ID
uint64    QueryPerformanceCounter converted to nanoseconds
uint32    width
uint32    height
uint32    MJPEG payload length
32 bytes  SHA-256 of the payload
N bytes   native MJPEG payload
```

The helper accepts `STOP\n` on stdin. It calls `IMFSourceReader::Flush`, exits the
read loop, releases the source, and returns within 500 ms. EOF or any identity,
mode, control, sample, hash, write, or cancellation error is terminal. The Python
consumer independently checks readiness, every hash, frame continuity, monotonic
timestamps, geometry, read timeout, and process cancellation.
