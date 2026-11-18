RER U3 Mainline 1 single-shot live diagnostic, candidate baseline 9a14495.

This packet runs the already implemented Mainline 1 chain from the real USB
camera. It is side-by-side: it does not install a release, stop or replace the
existing field service, open the field state database, or write match/score data.

Before running, keep only the RER U3 camera connected and close AMCap, Windows
Camera, OBS, and browser camera pages. Keep the camera and table fixed.

Run run.ps1. Model loading may take a few minutes. Do not strike until the console
prints LIVE_RUNTIME_READY. Then make one clear shot. The run stops automatically
after 600 processed frames and prints the SEND_BACK ZIP path.

SHOT_PUBLISHED proves the live chain produced an artifact. SHOT_ABSTAINED is also
a valid fail-closed chain outcome for this development-only identity head. This
packet is not production acceptance and must not be described as production ready.
