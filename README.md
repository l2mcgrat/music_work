# music_work
All my music projects stored on the cloud for free.

## WAV visualizer

The script [music_visualizer.py](music_visualizer.py) can visualize `.wav` files as a waveform and/or spectrum.

Run:
- `./.venv/Scripts/python.exe ./music_visualizer.py "path\to\audio.wav"`

Useful options:
- `--mode both|waveform|spectrum`
- `--mono` (mix down to mono)
- `--db` (spectrum in dB)
- `--play` (plays audio while visualizing)
- `--start 30 --duration 10` (visualize a segment)

Example:
- `./.venv/Scripts/python.exe ./music_visualizer.py "song.wav" --mode both --mono --db --play`

Export a file instead of showing a window:
- `./.venv/Scripts/python.exe ./music_visualizer.py "song.wav" --no-gui --export-auto --out-dir "./visualizer_exports" --export-format png --db --mono`

Logic-style full-track waveform in the export:
- `./.venv/Scripts/python.exe ./music_visualizer.py "song.wav" --waveform-view full --no-gui --export-auto --out-dir "./visualizer_exports" --export-format mp4 --overwrite-export --export-full`

Preview + export (shows the live window first, then saves an MP4 after you close it):
- `./.venv/Scripts/python.exe ./music_visualizer.py "song.wav" --mode both --db --play --export-auto --out-dir "./visualizer_exports" --export-format mp4 --export-seconds 15 --preview-export`

Batch export (folder -> one output folder):
- `powershell -ExecutionPolicy Bypass -File .\export_visualizations.ps1 -SourceDir ".\Seven_Deadly_Sins2_v2" -OutDir ".\visualizer_exports\Seven_Deadly_Sins2_v2" -Format png -UseDb -Mono -Seconds 5`

Batch export with full-track waveform:
- `powershell -ExecutionPolicy Bypass -File .\export_visualizations.ps1 -SourceDir ".\Seven_Deadly_Sins2_v2" -OutDir ".\visualizer_exports\Seven_Deadly_Sins2_v2" -Format mp4 -WaveformView full -Overwrite -Full`

## Batch convert audio to WAV

There’s a helper script [convert_audio_to_wav.ps1](convert_audio_to_wav.ps1) that converts common audio formats (and also fixes files that were renamed to `.wav` but aren’t real WAVs) into standard PCM `.wav` files.

Example (convert everything under `Seven_Deadly_Sins2_v1` into a `converted_wav` folder):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir ".\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100`

Convert into the same folder (creates `*_converted.wav` next to each file):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir ".\Seven_Deadly_Sins2_v1" -OutDir ".\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100`

Dry run (prints what it would do):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir ".\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100 -DryRun`
