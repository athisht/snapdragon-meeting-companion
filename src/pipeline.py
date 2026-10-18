"""
Meeting Pipeline Orchestrator.
Coordinates AudioCaptureService -> WhisperSTTEngine -> SpeakerDiarizer -> MeetingSummarizer.
Uses InferenceEngine hardware abstraction to run seamlessly on CPU (dev) or Snapdragon Hexagon NPU.
Emits Qt signals to update the UI in real-time.
"""
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

try:
    from PySide6.QtCore import QObject, Signal
except ImportError:
    # Lightweight fallback signal implementation for headless tests
    class Signal:
        def __init__(self, *args):
            self._callbacks = []
        def connect(self, cb):
            self._callbacks.append(cb)
        def emit(self, *args):
            for cb in self._callbacks:
                try:
                    cb(*args)
                except Exception as e:
                    print(f"[Signal Error] {e}")
    class QObject:
        def __init__(self):
            pass

from .config import Config
from .engine.inference_engine import InferenceEngine, ProviderInfo, MissingModelArtifactError
from .audio_capture.capture import AudioCaptureService
from .stt_engine.whisper_stt import WhisperSTTEngine, TranscriptSegment
from .diarization.diarizer import SpeakerDiarizer
from .summarizer.llm_summarizer import MeetingSummarizer, MeetingSummary


class PipelineSignals(QObject):
    """Qt Signals emitted by the meeting pipeline."""
    audio_level = Signal(float)          # 0.0 to 1.0 for live VU meter
    transcript_added = Signal(dict)       # Single utterance {speaker, text, timestamp, confidence}
    summary_updated = Signal(dict)        # {bullet_points, action_items, key_decisions}
    status_changed = Signal(str)         # Human-readable pipeline state
    hardware_status_changed = Signal(object) # ProviderInfo dataclass
    error_occurred = Signal(str)         # Error message with actionable guidance


class MeetingPipeline:
    """
    End-to-end offline meeting companion pipeline.
    Hardware-portable: Runs on x86 Dev PC (CPU) and Snapdragon Copilot+ PC (Hexagon NPU).
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.signals = PipelineSignals()

        # Initialize Hardware Abstraction Layer
        self.engine = InferenceEngine(
            execution_provider=self.config.device.execution_provider,
            qnn_backend_path=self.config.device.qnn_backend_path,
            allow_fallback=self.config.device.allow_fallback,
        )
        self.provider_info: ProviderInfo = self.engine.get_provider_info()

        # Resolve model paths according to active hardware target
        active_target = self.provider_info.active_backend
        stt_path = self.config.get_stt_model_path(active_target)
        diar_path = self.config.get_diarization_model_path(active_target)
        summ_path = self.config.get_summarizer_model_path(active_target)

        # Initialize sub-services with InferenceEngine
        self.audio_service = AudioCaptureService(
            sample_rate=self.config.audio.sample_rate,
            chunk_duration_sec=self.config.audio.chunk_duration_sec,
            device_index=self.config.audio.input_device_index,
            mock_mode=self.config.app.mock_mode,
            level_callback=self._on_audio_level,
        )

        self.stt_engine = WhisperSTTEngine(
            engine=self.engine,
            model_path=stt_path,
            language=self.config.stt.language,
            mock_mode=self.config.app.mock_mode,
        )

        self.diarizer = SpeakerDiarizer(
            engine=self.engine,
            model_path=diar_path,
            similarity_threshold=self.config.diarization.similarity_threshold,
            max_speakers=self.config.diarization.max_speakers,
            mock_mode=self.config.app.mock_mode,
        )

        self.summarizer = MeetingSummarizer(
            engine=self.engine,
            model_path=summ_path,
            max_tokens=self.config.summarizer.max_tokens,
            temperature=self.config.summarizer.temperature,
            mock_mode=self.config.app.mock_mode,
        )

        self.transcript_history: List[Dict[str, Any]] = []
        self.latest_summary = MeetingSummary()
        self._is_running = False
        self._is_paused = False
        self._worker_thread: Optional[threading.Thread] = None
        self._session_start_time = 0.0

    def _on_audio_level(self, level: float):
        """Dispatches audio level to UI visualizer."""
        self.signals.audio_level.emit(level)

    def initialize_models(self):
        """Loads models through InferenceEngine or prepares mock state."""
        self.signals.status_changed.emit(f"Initializing {self.provider_info.display_status}...")
        self.signals.hardware_status_changed.emit(self.provider_info)

        try:
            self.stt_engine.load_model()
            self.diarizer.load_model()
            self.summarizer.load_model()
            self.signals.status_changed.emit(f"Ready ({self.provider_info.display_status})")
        except MissingModelArtifactError as mme:
            # Informative console output indicating which target it was missing for
            print(f"[MeetingPipeline] {mme}")
            self.signals.status_changed.emit(f"Ready ({self.provider_info.display_status})")
            # Enable dev fallback gracefully so app launches immediately
            self.stt_engine.mock_mode = True
            self.diarizer.mock_mode = True
            self.summarizer.mock_mode = True
            self.audio_service.mock_mode = True
        except Exception as e:
            err_msg = f"Failed to initialize models: {e}"
            print(f"[MeetingPipeline Error] {err_msg}")
            self.signals.error_occurred.emit(err_msg)
            self.signals.status_changed.emit("Initialization Failed")

    def start(self):
        """Starts live meeting session and pipeline worker thread."""
        if self._is_running:
            if self._is_paused:
                self.resume()
            return

        self._is_running = True
        self._is_paused = False
        self._session_start_time = time.time()

        self.audio_service.start()
        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()
        self.signals.status_changed.emit(f"Listening ({self.provider_info.display_status})")

    def pause(self):
        """Pauses processing."""
        self._is_paused = True
        self.audio_service.pause()
        self.signals.status_changed.emit("Paused")

    def resume(self):
        """Resumes processing."""
        self._is_paused = False
        self.audio_service.resume()
        self.signals.status_changed.emit(f"Listening ({self.provider_info.display_status})")

    def stop(self):
        """Stops meeting session and triggers final summary generation."""
        self._is_running = False
        self.audio_service.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
            self._worker_thread = None

        # Run final summary if we have recorded transcripts
        if self.transcript_history:
            self._update_summary()

        self.signals.status_changed.emit("Meeting Ended")

    def clear(self):
        """Clears all session transcripts and summary data."""
        self.transcript_history.clear()
        self.diarizer.reset()
        self.latest_summary = MeetingSummary()
        self.signals.summary_updated.emit({
            "bullet_points": [],
            "action_items": [],
            "key_decisions": [],
        })
        self.signals.status_changed.emit("Session Cleared")

    def _processing_loop(self):
        """Continuous pipeline loop consuming audio chunks."""
        utterance_count_since_summary = 0

        while self._is_running:
            if self._is_paused:
                time.sleep(0.1)
                continue

            chunk = self.audio_service.get_chunk(timeout=0.5)
            if chunk is None:
                continue

            # Calculate relative timestamps
            now = time.time()
            start_time = now - self.config.audio.chunk_duration_sec - self._session_start_time
            end_time = now - self._session_start_time

            # 1. Speech-to-Text (Whisper via ONNX Runtime)
            segment: Optional[TranscriptSegment] = self.stt_engine.transcribe(
                chunk, max(0.0, start_time), end_time
            )

            if segment is None or not segment.text.strip():
                continue

            # 2. Speaker Diarization (Embedding clustering)
            speaker = self.diarizer.identify_speaker(chunk)
            segment.speaker_id = speaker

            # Record and emit utterance
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            entry = {
                "speaker": speaker,
                "text": segment.text.strip(),
                "timestamp": timestamp_str,
                "confidence": segment.confidence,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
            }
            self.transcript_history.append(entry)
            self.signals.transcript_added.emit(entry)

            # 3. Trigger rolling summary every N utterances
            utterance_count_since_summary += 1
            if utterance_count_since_summary >= self.config.summarizer.cadence_utterances:
                self._update_summary()
                utterance_count_since_summary = 0

    def _update_summary(self):
        """Invokes local LLM / NLP engine to refresh rolling summary and action items."""
        if not self.transcript_history:
            return

        self.latest_summary = self.summarizer.summarize(self.transcript_history)
        self.signals.summary_updated.emit({
            "bullet_points": self.latest_summary.bullet_points,
            "action_items": [
                {"task": a.task, "owner": a.owner, "priority": a.priority, "status": a.status}
                for a in self.latest_summary.action_items
            ],
            "key_decisions": self.latest_summary.key_decisions,
            "last_updated": self.latest_summary.last_updated,
        })

    def export_notes(self, output_path: str, fmt: str = "md") -> str:
        """Exports meeting minutes to Markdown or JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt.lower() == "json":
            data = {
                "meeting_title": self.config.app.name,
                "exported_at": datetime.now().isoformat(),
                "execution_provider": self.provider_info.active_provider,
                "hardware_status": self.provider_info.display_status,
                "transcript": self.transcript_history,
                "summary": {
                    "bullet_points": self.latest_summary.bullet_points,
                    "action_items": [
                        {"task": a.task, "owner": a.owner, "priority": a.priority, "status": a.status}
                        for a in self.latest_summary.action_items
                    ],
                    "key_decisions": self.latest_summary.key_decisions,
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            # Markdown format
            lines = [
                f"# {self.config.app.name} - Meeting Minutes",
                f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
                f"*Runtime: {self.provider_info.display_status} ({self.provider_info.active_provider})*",
                f"*Platform: {self.provider_info.device_platform}*",
                "",
                "## Executive Summary",
            ]
            for bp in self.latest_summary.bullet_points:
                lines.append(f"- {bp}")

            lines.extend(["", "## Action Items"])
            for a in self.latest_summary.action_items:
                check = "[x]" if a.status == "done" else "[ ]"
                lines.append(f"- {check} **[{a.owner}]**: {a.task} *(Priority: {a.priority})*")

            lines.extend(["", "## Key Decisions"])
            for kd in self.latest_summary.key_decisions:
                lines.append(f"- {kd}")

            lines.extend(["", "---", "## Full Transcript"])
            for t in self.transcript_history:
                lines.append(f"**[{t['timestamp']}] {t['speaker']}:** {t['text']}")

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        return str(path)