# Mobile toolchain

This repository pins Flutter **3.44.7** on the **stable** channel.  Install it
outside the repository so SDK binaries and download caches are never tracked:

```sh
scripts/bootstrap_flutter.sh "${HOME}/Library/Caches/billiards-ai/flutter"
```

The approved target is macOS arm64. The script checks the official Flutter
release JSON against `flutter_release_manifest.json`, verifies the official
release revision (`hash`) separately, then requires the official JSON
`sha256` and the downloaded archive's SHA-256 to match exactly before
extraction. The manifest is immutable during bootstrap. It prints the resulting Flutter
binary path; use that path's parent directory when adding Flutter to `PATH`.

Bootstrap verifies the cached archive on every run and rebuilds the SDK from it,
so a modified cached executable is never trusted. Only one bootstrap may use a
cache at a time; an existing lock, including a stale lock, fails closed.

`toolchain.json` records the machine-readable output from the approved
bootstrap invocation for the development host.

## App shell

The Flutter project intentionally targets only iOS and Android. It contains a
safe live-coach placeholder; media, networking, storage, permissions and
platform adapters are introduced only in later reviewed phases.

After bootstrapping the pinned SDK, run the local gate from this directory:

```sh
flutter pub get
dart format --output=none --set-exit-if-changed .
flutter analyze
flutter test
```
