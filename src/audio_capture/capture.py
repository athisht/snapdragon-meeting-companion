"""
Audio Capture Module
Captures microphone audio at 16kHz mono (required by Whisper STT).
Supports real microphone streaming via sounddevice or mock synthetic stream for testing.
"""
import time
import math
import queue
import threading
from typing import Optional, Callable
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class AudioCaptureService:
    """
    Microphone audio capture service.
    
    Streams live 16,000 Hz, 1-channel (mono) float32 audio chunks into a queue.
    Provides audio energy level callbacks for live UI VU meters.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_sec: float = 4.0,
        device_index: Optional[int] = None,
        mock_mode: bool = False,
        level_callback: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.device_index = device_index
        self.mock_mode = mock_mode
        self.level_callback = level_callback

        self.chunk_samples = int(self.sample_rate * self.chunk_duration_sec)
        self.audio_queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._stream = None

    def start(self):
        """Starts audio capture thread or stream."""
        if self._is_running:
            return
        self._is_running = True
        self._is_paused = False

        if self.mock_mode or not SOUNDDEVICE_AVAILABLE:
            if not SOUNDDEVICE_AVAILABLE and not self.mock_mode:
                print("[AudioCapture] 'sounddevice' not installed. Falling back to synthetic audio mock.")
            self._thread = threading.Thread(target=self._mock_audio_loop, daemon=True)
            self._thread.start()
        else:
            self._start_hardware_stream()

    def _start_hardware_stream(self):
        """Initializes and starts real microphone capture using sounddevice."""
        try:
            buffer = []
            samples_collected = 0

            def audio_callback(indata, frames, time_info, status):
                if not self._is_running or self._is_paused:
                    return
                # Mono float32 slice
                mono_data = indata[:, 0].copy()
                
                # Compute RMS audio level for UI meter (0.0 to 1.0)
                rms = float(np.sqrt(np.mean(mono_data**2)))
                level = min(1.0, rms * 8.0) # boost for visibility
                if self.level_callback:
                    self.level_callback(level)

                nonlocal buffer, samples_collected
                buffer.append(mono_data)
                samples_collected += len(mono_data)

                if samples_collected >= self.chunk_samples:
                    chunk = np.concatenate(buffer)[:self.chunk_samples]
                    self.audio_queue.put(chunk)
                    buffer = []
                    samples_collected = 0

            # Blocksize 1024 (~64ms buffer for responsive level metering)
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device_index,
                blocksize=1024,
                callback=audio_callback,
            )
            self._stream.start()
            print(f"[AudioCapture] Microphone stream started ({self.sample_rate}Hz mono)")
        except Exception as e:
            print(f"[AudioCapture] Failed to open microphone device: {e}. Switching to mock audio.")
            self.mock_mode = True
            self._thread = threading.Thread(target=self._mock_audio_loop, daemon=True)
            self._thread.start()

    def _mock_audio_loop(self):
        """Generates realistic synthetic audio pulses for offline testing without mic."""
        print("[AudioCapture] Mock audio generator active.")
        t = 0.0
        while self._is_running:
            if self._is_paused:
                time.sleep(0.1)
                continue

            # Simulate 100ms mini-frames for the VU meter
            frame_samples = int(self.sample_rate * 0.1)
            t_axis = np.linspace(t, t + 0.1, frame_samples, endpoint=False)
            t += 0.1

            # Rhythmic speech-like amplitude modulation
            envelope = max(0.0, math.sin(t * 1.5) * math.cos(t * 0.7))
            noise = np.random.normal(0, 0.05, frame_samples).astype(np.float32)
            carrier = (0.2 * np.sin(2 * np.pi * 320 * t_axis)).astype(np.float32)
            mock_frame = (envelope * (carrier + noise)).astype(np.float32)

            # Emit level for UI
            rms = float(np.sqrt(np.mean(mock_frame**2)))
            level = min(1.0, rms * 5.0)
            if self.level_callback:
                self.level_callback(level)

            # Accumulate and put full chunk every chunk_duration_sec
            if int(t / self.chunk_duration_sec) > int((t - 0.1) / self.chunk_duration_sec):
                dummy_chunk = np.random.normal(0, 0.1, self.chunk_samples).astype(np.float32)
                self.audio_queue.put(dummy_chunk)

            time.sleep(0.1)

    def pause(self):
        """Pauses recording."""
        self._is_paused = True

    def resume(self):
        """Resumes recording."""
        self._is_paused = False

    def stop(self):
        """Stops audio capture cleanly."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[AudioCapture] Audio capture stopped.")

    def get_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Retrieves next audio chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._is_running and not self._is_paused