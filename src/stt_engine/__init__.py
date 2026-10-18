"""Speech-To-Text module for Qualcomm AI Hub Whisper models."""
from .whisper_stt import WhisperSTTEngine, TranscriptSegment

__all__ = ["WhisperSTTEngine", "TranscriptSegment"]