"""Spoken driver alerts through the machine's speakers.

No third-party dependencies (the target machines may be offline): phrases
play from pre-recorded WAV files via the OS (winsound on Windows, aplay on
Linux), and fall back to a system text-to-speech engine (SAPI on Windows,
espeak on Linux) when the WAV files are missing.
"""

from __future__ import annotations

import heapq
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from .alerts import Alert, priority_for

# Alert key -> (spoken phrase, wav file name). The three drowsy stages
# share one phrase.
SPOKEN: dict[str, tuple[str, str]] = {
    "smoking": ("Stop smoking", "stop_smoking.wav"),
    "phone_use": ("Do not use phone", "do_not_use_phone.wav"),
    "drowsy_microsleep": ("Please take a break", "please_take_a_break.wav"),
    "drowsy_perclos": ("Please take a break", "please_take_a_break.wav"),
    "drowsy_yawning": ("Please take a break", "please_take_a_break.wav"),
}

# Confirmation phrase played once at startup so the speaker path is proven
# working before any alert is needed.
_CHIME = ("Driver monitoring active", "system_on.wav")


def phrase_for(alert: Alert) -> str | None:
    """Spoken line for an alert, or None when it should stay silent."""
    entry = SPOKEN.get(alert.key)
    return entry[0] if entry is not None else None


def _pick_wav_player(device: str = "") -> Callable[[Path], None] | None:
    if sys.platform == "win32":
        try:
            import winsound

            def play(path: Path) -> None:
                # SND_SYNC: the worker thread blocks until the phrase
                # finishes, so every word is heard before the next one.
                winsound.PlaySound(
                    str(path),
                    winsound.SND_FILENAME | winsound.SND_NODEFAULT,
                )

            return play
        except ImportError:
            return None
    if shutil.which("aplay"):
        device_args = ["-D", device] if device else []

        def play(path: Path) -> None:
            # Keep stderr: a wrong/blocked ALSA device is otherwise an
            # invisible failure and the speaker simply stays silent.
            result = subprocess.run(
                ["aplay", "-q", *device_args, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                detail = (result.stderr or "").strip().splitlines()
                raise RuntimeError(detail[-1] if detail else f"aplay exited {result.returncode}")

        return play
    return None


def _pick_tts() -> Callable[[str], None] | None:
    if sys.platform == "win32":
        if not shutil.which("powershell"):
            return None

        def speak(text: str) -> None:
            # '' is PowerShell's escape for a literal single quote.
            quoted = text.replace("'", "''")
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "try { $s.SelectVoice('Microsoft Zira Desktop') } catch {}; "
                    f"$s.Speak('{quoted}')",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        return speak
    for tool in ("espeak-ng", "espeak"):
        found = shutil.which(tool)
        if found:

            def speak(text: str, tool: str = found) -> None:
                subprocess.Popen(
                    [tool, "-v", "en+f3", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            return speak
    return None


class PriorityPlaylist:
    """Pending WAV phrases ordered by alert priority (lowest number first).

    Ties keep arrival order. If the backlog is full, the least urgent phrase
    is dropped rather than a more important one that arrived later.
    """

    def __init__(self, maxsize: int = 6) -> None:
        self.maxsize = max(1, int(maxsize))
        self._items: list[tuple[int, int, Path]] = []
        self._seq = 0
        self._cond = threading.Condition()
        self._closed = False

    def put(self, path: Path, priority: int = 100) -> None:
        with self._cond:
            if self._closed:
                return
            heapq.heappush(self._items, (int(priority), self._seq, Path(path)))
            self._seq += 1
            while len(self._items) > self.maxsize:
                worst = max(self._items, key=lambda item: (item[0], item[1]))
                self._items.remove(worst)
                heapq.heapify(self._items)
            self._cond.notify()

    def get(self) -> Path | None:
        """Block for the next phrase; None means the playlist was closed."""
        with self._cond:
            while not self._items and not self._closed:
                self._cond.wait()
            if not self._items:
                return None
            return heapq.heappop(self._items)[2]

    def get_nowait(self) -> Path | None:
        with self._cond:
            if not self._items:
                return None
            return heapq.heappop(self._items)[2]

    def flush(self) -> None:
        with self._cond:
            self._items.clear()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._items.clear()
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


class _WavePlayer:
    """Serializes WAV playback on a worker thread.

    Playing synchronously on a dedicated thread means every phrase runs to
    completion and alerts that land close together queue back-to-back,
    instead of the OS cancelling the sound in progress. Pending phrases are
    held in a priority queue, so a sleep warning goes next even if a phone
    phrase arrived first. A phrase already being spoken is not interrupted.
    """

    def __init__(self, play: Callable[[Path], None]) -> None:
        self._play = play
        self._queue = PriorityPlaylist()
        self._thread = threading.Thread(
            target=self._run, name="drivermon-audio", daemon=True
        )
        self._thread.start()

    def play(self, path: Path, priority: int = 100) -> None:
        self._queue.put(path, priority)

    def flush(self) -> None:
        """Drop backlog phrases that have not started playing yet.

        The phrase already playing is left alone: stopping it mid-word is
        not possible with the sync backends, and one trailing phrase after
        a condition clears is acceptable ("stop with one extra").
        """
        self._queue.flush()

    def close(self) -> None:
        self._queue.close()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            path = self._queue.get()
            if path is None:
                return
            try:
                self._play(path)
            except RuntimeError as exc:
                print(f"audio: could not play {path}: {exc}", flush=True)


class AudioAlerter:
    """AlertManager sink that speaks every alert it receives.

    Repeat pacing is the AlertManager's job (it re-alerts each cooldown
    while a condition stays active); this sink speaks every one of them so
    the driver keeps hearing the warning until the condition clears.
    """

    def __init__(
        self,
        wav_dir: str | Path = "data/audio",
        enabled: bool = True,
        device: str = "",
    ):
        self.enabled = enabled
        self._wav_dir = Path(wav_dir)
        self._play_wav = _pick_wav_player(device)
        self._tts = _pick_tts()
        self._player = _WavePlayer(self._play_wav) if self._play_wav is not None else None
        self._noted_unavailable = False

    @property
    def available(self) -> bool:
        return self._play_wav is not None or self._tts is not None

    def __call__(self, alert: Alert) -> None:
        if not self.enabled:
            return
        phrase = phrase_for(alert)
        if phrase is not None:
            self.speak(phrase, priority=priority_for(alert.key))

    def speak(self, phrase: str, priority: int = 100) -> None:
        filename = next(
            (name for text, name in SPOKEN.values() if text == phrase), None
        )
        if filename is None:
            return
        wav = self._wav_dir / filename
        if self._player is not None and wav.exists():
            print(f"speaking: {phrase}", flush=True)
            self._player.play(wav, priority)
            return
        if self._tts is not None:
            print(f"speaking: {phrase}", flush=True)
            self._tts(phrase)
            return
        if not self._noted_unavailable:
            print(
                "note: no audio backend (winsound/aplay/espeak/SAPI) and no "
                f"WAV phrases in {self._wav_dir} - voice alerts stay silent",
                flush=True,
            )
            self._noted_unavailable = True

    def flush(self) -> None:
        """Drop queued phrases: called when every condition has cleared so
        stale warnings do not keep playing after the object is gone."""
        if self._player is not None:
            self._player.flush()

    def close(self) -> None:
        if self._player is not None:
            self._player.close()

    def announce_startup(self) -> None:
        """Play the confirmation phrase once at launch: proves the speaker
        path works before the first alert is needed."""
        if not self.enabled:
            return
        text, filename = _CHIME
        wav = self._wav_dir / filename
        if self._player is not None and wav.exists():
            print(f"speaking: {text}", flush=True)
            self._player.play(wav)
            return
        if self._tts is not None:
            self._tts(text)
