# music_work
All my music projects stored on the cloud for free.

## WAV visualizer

The script [music_visualizer.py](music_visualizer.py) can visualize `.wav` files as a waveform and/or spectrum.

Run:
- `C:/Users/ve037081/Liam/music_work/.venv/Scripts/python.exe music_visualizer.py "path\to\audio.wav"`

Useful options:
- `--mode both|waveform|spectrum`
- `--mono` (mix down to mono)
- `--db` (spectrum in dB)
- `--play` (plays audio while visualizing)
- `--start 30 --duration 10` (visualize a segment)

Example:
- `C:/Users/ve037081/Liam/music_work/.venv/Scripts/python.exe music_visualizer.py "song.wav" --mode both --mono --db --play`

## Batch convert audio to WAV

There’s a helper script [convert_audio_to_wav.ps1](convert_audio_to_wav.ps1) that converts common audio formats (and also fixes files that were renamed to `.wav` but aren’t real WAVs) into standard PCM `.wav` files.

Example (convert everything under `Seven_Deadly_Sins2_v1` into a `converted_wav` folder):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir "C:\Users\ve037081\Liam\music_work\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100`

Convert into the same folder (creates `*_converted.wav` next to each file):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir "C:\Users\ve037081\Liam\music_work\Seven_Deadly_Sins2_v1" -OutDir "C:\Users\ve037081\Liam\music_work\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100`

Dry run (prints what it would do):
- `powershell -ExecutionPolicy Bypass -File .\convert_audio_to_wav.ps1 -SourceDir "C:\Users\ve037081\Liam\music_work\Seven_Deadly_Sins2_v1" -Mono -SampleRate 44100 -DryRun`
