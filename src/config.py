"""
Configuration manager for Snapdragon Offline Meeting Companion.
Loads settings from YAML and environment variables with structured dataclasses and sensible defaults.
Supports hardware-portable target-separated models (/models/cpu vs /models/qnn).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


def _load_env_file(filepath: Path = Path(".env")):
    """Simple parser for local .env without requiring extra external dependencies."""
    if not filepath.is_file():
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key not in os.environ:
                os.environ[key] = val


# Load any .env in project root if present
_load_env_file()


@dataclass
class AppConfig:
    name: str = "Snapdragon Offline Meeting Companion"
    version: str = "1.0.0"
    mock_mode: bool = False
    export_dir: str = "exports"


@dataclass
class DeviceConfig:
    execution_provider: str = "cpu"  # 'cpu', 'qnn', or 'cuda'
    allow_fallback: bool = True
    qnn_backend_path: str = "QnnHtp.dll"

    @property
    def target(self) -> str:
        """Alias for hardware target name."""
        return "hexagon_npu" if self.execution_provider == "qnn" else self.execution_provider

    @property
    def onnx_execution_provider(self) -> str:
        """Name of the primary ONNX Runtime execution provider string."""
        if self.execution_provider == "qnn":
            return "QNNExecutionProvider"
        if self.execution_provider == "cuda":
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_sec: float = 4.0
    silence_threshold: float = 0.01
    input_device_index: Optional[int] = None


@dataclass
class TargetModelPaths:
    stt_model_path: str = ""
    diarization_model_path: str = ""
    summarizer_model_path: str = ""


@dataclass
class STTModelConfig:
    language: str = "en"
    temperature: float = 0.0
    # Backwards compatibility attributes
    model_name: str = "whisper_base_en"
    model_path: str = "models/cpu/stt/whisper_base_en.onnx"


@dataclass
class DiarizationModelConfig:
    similarity_threshold: float = 0.72
    max_speakers: int = 6
    # Backwards compatibility attributes
    model_name: str = "speaker_embedding"
    model_path: str = "models/cpu/diarization/speaker_embedding.onnx"


@dataclass
class SummarizerModelConfig:
    max_tokens: int = 512
    temperature: float = 0.2
    cadence_utterances: int = 5
    # Backwards compatibility attributes
    model_name: str = "llama_3_2_1b"
    model_path: str = "models/cpu/llm/llama_3_2_1b"


@dataclass
class UIConfig:
    theme: str = "dark"
    window_title: str = "Snapdragon Meeting Companion (Offline Copilot+ PC)"
    width: int = 1240
    height: int = 820


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    models_cpu: TargetModelPaths = field(
        default_factory=lambda: TargetModelPaths(
            stt_model_path="models/cpu/stt/whisper_base_en.onnx",
            diarization_model_path="models/cpu/diarization/speaker_embedding.onnx",
            summarizer_model_path="models/cpu/llm/llama_3_2_1b",
        )
    )
    models_qnn: TargetModelPaths = field(
        default_factory=lambda: TargetModelPaths(
            stt_model_path="models/qnn/stt/whisper_base_en_quantized_qnn.onnx",
            diarization_model_path="models/qnn/diarization/speaker_embedding_qnn.onnx",
            summarizer_model_path="models/qnn/llm/llama_3_2_1b_qnn",
        )
    )
    stt: STTModelConfig = field(default_factory=STTModelConfig)
    diarization: DiarizationModelConfig = field(default_factory=DiarizationModelConfig)
    summarizer: SummarizerModelConfig = field(default_factory=SummarizerModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    def get_stt_model_path(self, target: Optional[str] = None) -> str:
        tgt = (target or self.device.execution_provider).lower()
        if tgt in ("qnn", "hexagon", "npu"):
            return self.models_qnn.stt_model_path
        return self.models_cpu.stt_model_path

    def get_diarization_model_path(self, target: Optional[str] = None) -> str:
        tgt = (target or self.device.execution_provider).lower()
        if tgt in ("qnn", "hexagon", "npu"):
            return self.models_qnn.diarization_model_path
        return self.models_cpu.diarization_model_path

    def get_summarizer_model_path(self, target: Optional[str] = None) -> str:
        tgt = (target or self.device.execution_provider).lower()
        if tgt in ("qnn", "hexagon", "npu"):
            return self.models_qnn.summarizer_model_path
        return self.models_cpu.summarizer_model_path

    @classmethod
    def load(cls, config_path: str = "config/settings.yaml") -> "Config":
        path = Path(config_path)
        raw: Dict[str, Any] = {}
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            print(f"[Config] File not found at {path}. Using default configuration.")

        app_data = raw.get("app", {})
        device_data = raw.get("device", {})
        audio_data = raw.get("audio", {})
        models_data = raw.get("models", {})
        ui_data = raw.get("ui", {})

        # Environment variable override for EXECUTION_PROVIDER
        env_provider = os.environ.get("EXECUTION_PROVIDER")
        if env_provider:
            device_data["execution_provider"] = env_provider.strip().lower()

        # Parse target-separated model paths
        cpu_models = models_data.get("cpu", {})
        qnn_models = models_data.get("qnn", {})

        # Parse model parameters
        stt_data = models_data.get("stt", {})
        diar_data = models_data.get("diarization", {})
        summ_data = models_data.get("summarizer", {})

        device_config = DeviceConfig(**device_data)

        # Backwards compatibility: keep .model_path on stt, diar, summ
        tgt = device_config.execution_provider.lower()
        is_qnn = tgt in ("qnn", "hexagon", "npu")

        cpu_target_paths = TargetModelPaths(
            stt_model_path=cpu_models.get("stt_model_path", "models/cpu/stt/whisper_base_en.onnx"),
            diarization_model_path=cpu_models.get("diarization_model_path", "models/cpu/diarization/speaker_embedding.onnx"),
            summarizer_model_path=cpu_models.get("summarizer_model_path", "models/cpu/llm/llama_3_2_1b"),
        )
        qnn_target_paths = TargetModelPaths(
            stt_model_path=qnn_models.get("stt_model_path", "models/qnn/stt/whisper_base_en_quantized_qnn.onnx"),
            diarization_model_path=qnn_models.get("diarization_model_path", "models/qnn/diarization/speaker_embedding_qnn.onnx"),
            summarizer_model_path=qnn_models.get("summarizer_model_path", "models/qnn/llm/llama_3_2_1b_qnn"),
        )

        stt_conf = STTModelConfig(**stt_data)
        stt_conf.model_path = qnn_target_paths.stt_model_path if is_qnn else cpu_target_paths.stt_model_path

        diar_conf = DiarizationModelConfig(**diar_data)
        diar_conf.model_path = qnn_target_paths.diarization_model_path if is_qnn else cpu_target_paths.diarization_model_path

        summ_conf = SummarizerModelConfig(**summ_data)
        summ_conf.model_path = qnn_target_paths.summarizer_model_path if is_qnn else cpu_target_paths.summarizer_model_path

        return cls(
            app=AppConfig(**app_data),
            device=device_config,
            audio=AudioConfig(**audio_data),
            models_cpu=cpu_target_paths,
            models_qnn=qnn_target_paths,
            stt=stt_conf,
            diarization=diar_conf,
            summarizer=summ_conf,
            ui=UIConfig(**ui_data),
        )