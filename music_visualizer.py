"""WAV music visualizer.

Visualizes a .wav file as a scrolling waveform and/or spectrum.

Dependencies:
- numpy
- matplotlib
- sounddevice (optional, for playback)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WavData:
	sample_rate: int
	samples: np.ndarray  # shape: (n_samples, channels), float32 in [-1, 1]

	@property
	def n_samples(self) -> int:
		return int(self.samples.shape[0])

	@property
	def channels(self) -> int:
		return int(self.samples.shape[1])

	@property
	def duration_s(self) -> float:
		return float(self.n_samples) / float(self.sample_rate)


def _int24_bytes_to_int32(raw: bytes, n_channels: int) -> np.ndarray:
	"""Convert little-endian 24-bit PCM bytes to int32 array.

	Returns array shaped (n_frames, n_channels).
	"""

	if len(raw) % (3 * n_channels) != 0:
		raise ValueError("Invalid 24-bit PCM byte length")

	b = np.frombuffer(raw, dtype=np.uint8)
	frames = b.reshape(-1, n_channels, 3)
	values = (
		frames[:, :, 0].astype(np.int32)
		| (frames[:, :, 1].astype(np.int32) << 8)
		| (frames[:, :, 2].astype(np.int32) << 16)
	)
	sign_bit = 1 << 23
	values = (values ^ sign_bit) - sign_bit
	return values


def read_wav(path: Path) -> WavData:
	"""Read a PCM .wav and return float32 samples in [-1, 1]."""

	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(str(path))
	if path.suffix.lower() != ".wav":
		raise ValueError(f"Expected a .wav file, got: {path.name}")

	# Sniff header for clearer errors when a non-WAV file is renamed to .wav.
	with path.open("rb") as f:
		header = f.read(16)
	if len(header) < 12:
		raise ValueError("File is too small to be a valid WAV")

	is_riff_wave = header[0:4] == b"RIFF" and header[8:12] == b"WAVE"
	if not is_riff_wave:
		# Common non-WAV signatures
		if header[0:3] == b"ID3" or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
			raise ValueError(
				"This file is not a RIFF/WAVE .wav (it looks like an MP3 with an ID3 tag). "
				"Rename it to .mp3 (optional) and convert to WAV, e.g. with ffmpeg:\n"
				"  ffmpeg -i \"input.mp3\" -ac 1 -ar 44100 \"output.wav\"\n"
				"(ffmpeg can usually detect the format even if the extension is wrong.)"
			)
		if header[0:4] == b"fLaC":
			raise ValueError(
				"This file looks like FLAC, not WAV. Convert to WAV first (e.g. ffmpeg -i input.flac output.wav)."
			)
		if header[4:8] == b"ftyp":
			raise ValueError(
				"This file looks like an MP4/M4A container, not WAV. Convert to WAV first (e.g. ffmpeg -i input.m4a output.wav)."
			)
		if header[0:4] == b"RF64":
			raise ValueError(
				"This file is RF64 (extended WAV). The built-in WAV reader can't open it; convert to standard WAV (RIFF) first."
			)
		raise ValueError(
			"File does not start with a RIFF/WAVE header; it may not actually be a WAV file. "
			"Convert it to a standard PCM WAV (RIFF) and try again."
		)

	with wave.open(str(path), "rb") as wf:
		n_channels = wf.getnchannels()
		sampwidth = wf.getsampwidth()  # bytes per sample
		sample_rate = wf.getframerate()
		n_frames = wf.getnframes()

		if wf.getcomptype() != "NONE":
			raise ValueError(f"Unsupported WAV compression type: {wf.getcomptype()}")
		if sampwidth not in (1, 2, 3, 4):
			raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

		raw = wf.readframes(n_frames)

	if sampwidth == 1:
		# 8-bit PCM is unsigned
		arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
		arr = arr - 128
		max_int = 127
		arr = arr.reshape(-1, n_channels)
		samples = (arr.astype(np.float32) / float(max_int)).clip(-1.0, 1.0)
		return WavData(sample_rate=sample_rate, samples=samples)

	if sampwidth == 2:
		arr = np.frombuffer(raw, dtype="<i2").reshape(-1, n_channels)
		samples = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
		return WavData(sample_rate=sample_rate, samples=samples)

	if sampwidth == 3:
		arr = _int24_bytes_to_int32(raw, n_channels)
		samples = (arr.astype(np.float32) / float(1 << 23)).clip(-1.0, 1.0)
		return WavData(sample_rate=sample_rate, samples=samples)

	# sampwidth == 4
	arr = np.frombuffer(raw, dtype="<i4").reshape(-1, n_channels)
	samples = (arr.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
	return WavData(sample_rate=sample_rate, samples=samples)


def to_mono(samples: np.ndarray) -> np.ndarray:
	if samples.ndim != 2:
		raise ValueError("Expected samples shaped (n_samples, channels)")
	if samples.shape[1] == 1:
		return samples[:, 0]
	return samples.mean(axis=1)


def slice_audio(
	mono: np.ndarray,
	sample_rate: int,
	start_s: float = 0.0,
	duration_s: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
	start_idx = max(0, int(round(start_s * sample_rate)))
	end_idx = mono.shape[0] if duration_s is None else min(mono.shape[0], start_idx + int(round(duration_s * sample_rate)))
	if start_idx >= end_idx:
		raise ValueError("Selected segment is empty (check --start/--duration)")
	return mono[start_idx:end_idx], start_idx


def compute_spectrum(
	segment: np.ndarray,
	sample_rate: int,
	window: np.ndarray,
	db: bool,
	eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
	if segment.shape[0] != window.shape[0]:
		raise ValueError("segment/window length mismatch")

	x = segment.astype(np.float32) * window
	fft = np.fft.rfft(x)
	mag = np.abs(fft).astype(np.float32)
	freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / float(sample_rate)).astype(np.float32)
	if db:
		mag = 20.0 * np.log10(mag + eps)
	return freqs, mag


def visualize(
	wav: WavData,
	*,
	source_path: Optional[Path] = None,
	mono: bool,
	mode: str,
	play: bool,
	start_s: float,
	duration_s: Optional[float],
	waveform_ms: float,
	waveform_view: str,
	max_waveform_points: int,
	window_ms: float,
	hop_ms: float,
	fps: float,
	db: bool,
	title: Optional[str],
	export_path: Optional[Path] = None,
	export_seconds: Optional[float] = None,
	no_gui: bool = False,
	overwrite_export: bool = False,
	export_audio: bool = True,
) -> None:
	import matplotlib.pyplot as plt
	from matplotlib.animation import FuncAnimation

	audio = to_mono(wav.samples) if mono else wav.samples[:, 0]
	audio_seg, seg_start_idx = slice_audio(audio, wav.sample_rate, start_s=start_s, duration_s=duration_s)

	waveform_view = (waveform_view or "scroll").lower()
	if waveform_view not in ("scroll", "full"):
		raise ValueError("waveform_view must be 'scroll' or 'full'")

	waveform_len = max(16, int(round(wav.sample_rate * waveform_ms / 1000.0)))
	window_len = max(32, int(round(wav.sample_rate * window_ms / 1000.0)))
	hop_len = max(1, int(round(wav.sample_rate * hop_ms / 1000.0)))

	window = np.hanning(window_len).astype(np.float32)
	total_len = audio_seg.shape[0]

	nrows = 2 if mode == "both" else 1
	fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 6), constrained_layout=True)
	if nrows == 1:
		axes = [axes]

	waveform_ax = None
	spectrum_ax = None
	if mode in ("both", "waveform"):
		waveform_ax = axes[0]
	if mode in ("both", "spectrum"):
		spectrum_ax = axes[-1]

	shown_name = source_path.name if source_path is not None else ""
	start_title = title or f"{wav.sample_rate} Hz | {shown_name}".strip(" |")
	fig.suptitle(start_title if start_title else "WAV Visualizer")

	# Waveform setup
	waveform_line = None
	waveform_playhead = None
	if waveform_ax is not None:
		waveform_ax.set_ylim(-1.05, 1.05)
		waveform_ax.set_ylabel("Amplitude")
		waveform_ax.grid(True, alpha=0.2)

		if waveform_view == "full":
			seg_seconds = float(total_len) / float(wav.sample_rate)
			max_points = max(200, int(max_waveform_points))
			if total_len <= max_points:
				t_bins = (np.arange(total_len, dtype=np.float32) / float(wav.sample_rate)).astype(np.float32)
				mins = audio_seg.astype(np.float32, copy=False)
				maxs = audio_seg.astype(np.float32, copy=False)
			else:
				step = int(math.ceil(float(total_len) / float(max_points)))
				n_bins = int(math.ceil(float(total_len) / float(step)))
				pad_len = (n_bins * step) - total_len
				padded = np.pad(audio_seg.astype(np.float32, copy=False), (0, pad_len), mode="constant")
				reshaped = padded.reshape(n_bins, step)
				mins = reshaped.min(axis=1)
				maxs = reshaped.max(axis=1)
				t_bins = (np.arange(n_bins, dtype=np.float32) * float(step) / float(wav.sample_rate)).astype(np.float32)

			waveform_ax.fill_between(t_bins, mins, maxs, color="C0", alpha=0.7, linewidth=0)
			waveform_ax.axhline(0.0, color="black", lw=0.8, alpha=0.2)
			waveform_playhead = waveform_ax.axvline(0.0, color="C3", lw=1.4)
			waveform_ax.set_xlim(0.0, seg_seconds)
			waveform_ax.set_xlabel("Time (s) (segment)")
		else:
			t = np.arange(waveform_len, dtype=np.float32) / float(wav.sample_rate)
			(waveform_line,) = waveform_ax.plot(t, np.zeros_like(t), lw=1)
			waveform_ax.set_xlim(0.0, float(waveform_len) / float(wav.sample_rate))
			waveform_ax.set_xlabel("Time (s)")

	# Spectrum setup
	spectrum_line = None
	spectrum_freqs = None
	if spectrum_ax is not None:
		freqs = np.fft.rfftfreq(window_len, d=1.0 / float(wav.sample_rate)).astype(np.float32)
		spectrum_freqs = freqs
		(spectrum_line,) = spectrum_ax.plot(freqs, np.zeros_like(freqs), lw=1)
		spectrum_ax.set_xlim(0.0, float(wav.sample_rate) / 2.0)
		if db:
			spectrum_ax.set_ylabel("Magnitude (dB)")
			spectrum_ax.set_ylim(-120.0, 20.0)
		else:
			spectrum_ax.set_ylabel("Magnitude")
			spectrum_ax.set_ylim(0.0, 50.0)
		spectrum_ax.set_xlabel("Frequency (Hz)")
		spectrum_ax.grid(True, alpha=0.2)

	# Playback (optional). Disabled during export.
	playback_started_at = None
	if play and export_path is None:
		try:
			import sounddevice as sd

			sd.play(audio_seg.astype(np.float32), wav.sample_rate)
			playback_started_at = time.perf_counter()
		except Exception as e:  # noqa: BLE001
			print(f"Playback disabled: {e}")
			play = False

	# Animation state
	t0 = time.perf_counter()

	exporting = export_path is not None

	def _current_index(frame_i: int) -> int:
		if play and playback_started_at is not None:
			elapsed = time.perf_counter() - playback_started_at
			return int(elapsed * wav.sample_rate)
		if exporting:
			# For exported video, align frames to time so audio mux stays in sync.
			return int((float(frame_i) / float(max(1e-6, fps))) * wav.sample_rate)
		# Non-playback: advance by hop on each frame
		return frame_i * hop_len

	def _get_windowed_segment(idx: int) -> np.ndarray:
		if idx < 0:
			idx = 0
		seg = audio_seg[idx : idx + window_len]
		if seg.shape[0] < window_len:
			pad = np.zeros(window_len - seg.shape[0], dtype=np.float32)
			seg = np.concatenate([seg.astype(np.float32, copy=False), pad])
		return seg.astype(np.float32, copy=False)

	def _get_waveform_segment(idx: int) -> np.ndarray:
		start = max(0, idx - waveform_len)
		seg = audio_seg[start:idx]
		if seg.shape[0] < waveform_len:
			pad = np.zeros(waveform_len - seg.shape[0], dtype=np.float32)
			seg = np.concatenate([pad, seg.astype(np.float32, copy=False)])
		return seg.astype(np.float32, copy=False)

	def update(frame_i: int):
		idx = _current_index(frame_i)
		if idx >= total_len:
			if exporting:
				idx = max(0, total_len - 1)
			else:
				return ()

		artists = []
		if waveform_view == "scroll" and waveform_line is not None:
			wseg = _get_waveform_segment(idx)
			waveform_line.set_ydata(wseg)
			artists.append(waveform_line)
		elif waveform_view == "full" and waveform_playhead is not None:
			seg_t = float(idx) / float(wav.sample_rate)
			waveform_playhead.set_xdata([seg_t, seg_t])
			artists.append(waveform_playhead)

		if spectrum_line is not None and spectrum_freqs is not None:
			sseg = _get_windowed_segment(idx)
			_, mag = compute_spectrum(sseg, wav.sample_rate, window, db=db)
			spectrum_line.set_ydata(mag)
			if not db:
				# Auto-scale gently for linear magnitudes
				ymax = float(np.percentile(mag, 99.0))
				ymax = max(1e-6, ymax)
				spectrum_ax.set_ylim(0.0, ymax * 1.2)
			artists.append(spectrum_line)

		# Update title with position
		now_s = (seg_start_idx + idx) / float(wav.sample_rate)
		fig.suptitle(f"{start_title or 'WAV Visualizer'}  |  t={now_s:0.2f}s")

		# Stop automatically if not playing and we reached end
		if not play and idx + window_len >= total_len:
			# Let the last frame draw, then close.
			if (time.perf_counter() - t0) > 0.5:
				plt.close(fig)

		return tuple(artists)

	interval_ms = 1000.0 / float(max(1.0, fps))

	if export_path is not None:
		export_path = Path(export_path)
		export_path.parent.mkdir(parents=True, exist_ok=True)

		if overwrite_export and export_path.exists():
			export_path.unlink()

		ext = export_path.suffix.lower()
		if ext in (".png", ".jpg", ".jpeg", ".webp"):
			# Save a single snapshot at the start position.
			update(0)
			fig.savefig(str(export_path), dpi=160)
			plt.close(fig)
			return

		if ext in (".mp4", ".gif"):
			# Save a short animation.
			seg_seconds = float(total_len) / float(wav.sample_rate)
			if export_seconds is None or float(export_seconds) <= 0:
				seconds = seg_seconds
			else:
				seconds = min(seg_seconds, float(export_seconds))
			seconds = max(0.25, float(seconds))
			frames = max(1, int(math.ceil(seconds * float(fps))))
			anim = FuncAnimation(
				fig,
				update,
				interval=interval_ms,
				blit=False,
				cache_frame_data=False,
				save_count=frames,
			)
			if ext == ".mp4":
				from matplotlib.animation import FFMpegWriter

				writer = FFMpegWriter(fps=float(fps), bitrate=1800)

				# Write silent video first, then mux audio (Matplotlib doesn't reliably embed audio).
				tmp_video_path = None
				tmp_audio_path = None
				try:
					with tempfile.NamedTemporaryFile(
						mode="wb",
						suffix=".mp4",
						prefix="viz_video_",
						dir=str(export_path.parent),
						delete=False,
					) as tmp:
						tmp_video_path = Path(tmp.name)

					anim.save(str(tmp_video_path), writer=writer, dpi=160)

					if export_audio:
						ffmpeg = shutil.which("ffmpeg")
						if not ffmpeg:
							raise RuntimeError("ffmpeg not found on PATH; cannot mux audio into MP4")

						audio_len = int(round(seconds * wav.sample_rate))
						audio_out = audio_seg[:audio_len].astype(np.float32, copy=False)
						pcm = np.clip(audio_out, -1.0, 1.0)
						pcm16 = (pcm * 32767.0).astype("<i2")

						with tempfile.NamedTemporaryFile(
							mode="wb",
							suffix=".wav",
							prefix="viz_audio_",
							dir=str(export_path.parent),
							delete=False,
						) as tmp:
							tmp_audio_path = Path(tmp.name)

						with wave.open(str(tmp_audio_path), "wb") as wf:
							wf.setnchannels(1)
							wf.setsampwidth(2)
							wf.setframerate(wav.sample_rate)
							wf.writeframes(pcm16.tobytes())

						mux = subprocess.run(
							[
								ffmpeg,
								"-y",
								"-i",
								str(tmp_video_path),
								"-i",
								str(tmp_audio_path),
								"-c:v",
								"copy",
								"-c:a",
								"aac",
								"-b:a",
								"192k",
								"-shortest",
								str(export_path),
							],
							capture_output=True,
							text=True,
							check=False,
						)
						if mux.returncode != 0:
							raise RuntimeError(f"ffmpeg mux failed: {mux.stderr.strip()}")
					else:
						tmp_video_path.replace(export_path)
				finally:
					if tmp_audio_path is not None and tmp_audio_path.exists():
						try:
							tmp_audio_path.unlink()
						except OSError:
							pass
					if tmp_video_path is not None and tmp_video_path.exists() and tmp_video_path != export_path:
						try:
							tmp_video_path.unlink()
						except OSError:
							pass
			else:
				# GIF requires pillow
				from matplotlib.animation import PillowWriter

				writer = PillowWriter(fps=float(fps))
				anim.save(str(export_path), writer=writer, dpi=120)

			plt.close(fig)
			return

		raise ValueError(f"Unsupported export extension: {ext} (use .png or .mp4 or .gif)")

	anim = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)

	if not no_gui:
		plt.show()
	else:
		plt.close(fig)

	if play:
		try:
			import sounddevice as sd

			sd.stop()
		except Exception:
			pass


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Visualize a .wav file (waveform and/or spectrum).")
	p.add_argument("wav", nargs="?", help="Path to a .wav file")
	p.add_argument("--mono", action="store_true", help="Mix down to mono")
	p.add_argument("--mode", choices=["both", "waveform", "spectrum"], default="both")
	p.add_argument("--play", action="store_true", help="Play audio while visualizing (requires sounddevice)")
	p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
	p.add_argument("--duration", type=float, default=None, help="Duration to visualize (seconds)")
	p.add_argument("--waveform-ms", type=float, default=250.0, help="Waveform window length (ms)")
	p.add_argument("--waveform-view", choices=["scroll", "full"], default="scroll", help="Waveform view: scrolling window or full-track overview")
	p.add_argument("--max-waveform-points", type=int, default=20000, help="Max points for full waveform overview (higher = more detail, slower)")
	p.add_argument("--window-ms", type=float, default=50.0, help="Spectrum FFT window length (ms)")
	p.add_argument("--hop-ms", type=float, default=20.0, help="Hop size between FFTs (ms)")
	p.add_argument("--fps", type=float, default=30.0, help="Animation frames per second")
	p.add_argument("--db", action="store_true", help="Show spectrum in dB")
	p.add_argument("--title", type=str, default=None, help="Custom title")
	p.add_argument("--export", type=str, default=None, help="Export to a file (.png/.mp4/.gif) instead of only showing")
	p.add_argument("--out-dir", type=str, default=None, help="Output directory (used with --export-auto)")
	p.add_argument(
		"--export-auto",
		action="store_true",
		help="Auto-name the export file from tags/filename. If --out-dir is omitted, writes to a 'visualizer_exports' folder next to the input wav.",
	)
	p.add_argument("--export-format", choices=["png", "mp4", "gif"], default="png", help="Export format for --export-auto")
	p.add_argument("--export-seconds", type=float, default=10.0, help="Seconds to export for mp4/gif")
	p.add_argument("--export-full", action="store_true", help="Export the full selected audio segment (ignores --export-seconds)")
	p.add_argument("--no-gui", action="store_true", help="Do not open a window (useful with --export)")
	p.add_argument("--overwrite-export", action="store_true", help="Overwrite export file if it already exists")
	p.add_argument(
		"--no-export-audio",
		dest="export_audio",
		action="store_false",
		default=True,
		help="Do not mux audio into exported MP4",
	)
	p.add_argument(
		"--preview-export",
		action="store_true",
		help="When exporting, show a live preview window first, then export after you close it",
	)
	return p


def _ffprobe_tags(path: Path) -> dict:
	"""Return tags via ffprobe when available (best-effort)."""

	try:
		result = subprocess.run(
			[
				"ffprobe",
				"-v",
				"error",
				"-print_format",
				"json",
				"-show_format",
				"-show_entries",
				"format_tags=title,artist,album",
				str(path),
			],
			capture_output=True,
			text=True,
			check=False,
		)
		if result.returncode != 0 or not result.stdout.strip():
			return {}
		import json

		obj = json.loads(result.stdout)
		tags = obj.get("format", {}).get("tags", {}) or {}
		# normalize keys to lower
		return {str(k).lower(): str(v) for k, v in tags.items() if v is not None}
	except Exception:
		return {}


def _suggest_song_stem(audio_path: Path) -> str:
	# Prefer embedded title tag if present.
	tags = _ffprobe_tags(audio_path)
	title = tags.get("title")
	if title:
		stem = title.strip()
	else:
		stem = audio_path.stem
		if stem.lower().endswith("_converted"):
			stem = stem[: -len("_converted")]
	# basic sanitization
	bad = '<>:"/\\|?*'
	for ch in bad:
		stem = stem.replace(ch, "_")
	stem = " ".join(stem.split()).strip(" .")
	return stem or audio_path.stem


def main(argv: Optional[list[str]] = None) -> int:
	args = build_parser().parse_args(argv)
	if not args.wav:
		print("Provide a .wav file path. Example: python music_visualizer.py path\\to\\song.wav")
		return 2

	wav_path = Path(args.wav)
	wav = read_wav(wav_path)

	export_path: Optional[Path] = None
	if args.export and args.export_auto:
		raise ValueError("Use either --export or --export-auto (not both)")
	if args.export:
		export_path = Path(args.export)
	elif args.export_auto:
		out_dir = Path(args.out_dir) if args.out_dir else (wav_path.parent / "visualizer_exports")
		out_dir.mkdir(parents=True, exist_ok=True)
		stem = _suggest_song_stem(wav_path)
		export_path = out_dir / f"{stem}.{args.export_format}"

	# If user requested preview+export, do preview first, then export.
	if export_path is not None and args.preview_export and not args.no_gui:
		visualize(
			wav,
			source_path=wav_path,
			mono=args.mono,
			mode=args.mode,
			play=bool(args.play),
			start_s=float(args.start),
			duration_s=None if args.duration is None else float(args.duration),
			waveform_ms=float(args.waveform_ms),
			waveform_view=str(args.waveform_view),
			max_waveform_points=int(args.max_waveform_points),
			window_ms=float(args.window_ms),
			hop_ms=float(args.hop_ms),
			fps=float(args.fps),
			db=args.db,
			title=args.title,
			export_path=None,
			export_seconds=None,
			no_gui=False,
			overwrite_export=False,
		)

		# Export pass (no playback, no GUI)
		visualize(
			wav,
			source_path=wav_path,
			mono=args.mono,
			mode=args.mode,
			play=False,
			start_s=float(args.start),
			duration_s=None if args.duration is None else float(args.duration),
			waveform_ms=float(args.waveform_ms),
			waveform_view=str(args.waveform_view),
			max_waveform_points=int(args.max_waveform_points),
			window_ms=float(args.window_ms),
			hop_ms=float(args.hop_ms),
			fps=float(args.fps),
			db=args.db,
			title=args.title,
			export_path=export_path,
			export_seconds=float(args.export_seconds),
			no_gui=True,
			overwrite_export=bool(args.overwrite_export),
			export_audio=bool(args.export_audio),
		)
		return 0

	# Basic validation
	if args.window_ms <= 0 or args.hop_ms <= 0 or args.waveform_ms <= 0:
		raise ValueError("--window-ms/--hop-ms/--waveform-ms must be positive")
	if args.fps <= 0:
		raise ValueError("--fps must be positive")

	visualize(
		wav,
		source_path=wav_path,
		mono=args.mono,
		mode=args.mode,
		play=bool(args.play),
		start_s=float(args.start),
		duration_s=None if args.duration is None else float(args.duration),
		waveform_ms=float(args.waveform_ms),
		waveform_view=str(args.waveform_view),
		max_waveform_points=int(args.max_waveform_points),
		window_ms=float(args.window_ms),
		hop_ms=float(args.hop_ms),
		fps=float(args.fps),
		db=args.db,
		title=args.title,
		export_path=export_path,
		export_seconds=None
		if export_path is None
		else (0.0 if bool(args.export_full) else float(args.export_seconds)),
		no_gui=bool(args.no_gui),
		overwrite_export=bool(args.overwrite_export),
		export_audio=bool(args.export_audio),
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())