"""
Hardware Abstraction Layer & ONNX Runtime Inference Engine.
Provides a unified interface across x86 Dev (CPU) and Snapdragon ARM64 (Hexagon NPU via QNN).
"""
import os
import platform
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False


class ExecutionBackend(str, Enum):
    CPU = "cpu"
    QNN = "qnn"      # Qualcomm Hexagon NPU (Snapdragon Copilot+ PCs)
    CUDA = "cuda"    # Optional GPU provider


class MissingModelArtifactError(FileNotFoundError):
    """Raised when a target-specific model file is missing with clear actionable guidance."""
    def __init__(self, model_type: str, target: str, path: Union[str, Path], suggestions: Optional[str] = None):
        self.model_type = model_type
        self.target = target
        self.path = str(path)
        msg = (
            f"\n[Missing Model Artifact]\n"
            f"  Target Hardware : {target.upper()}\n"
            f"  Component       : {model_type}\n"
            f"  Expected Path   : {self.path}\n\n"
            f"  How to resolve:\n"
        )
        if target.lower() == "qnn":
            msg += (
                f"  1. Ensure you are on a Snapdragon Copilot+ PC (ARM64 Windows).\n"
                f"  2. Export the compiled model using Qualcomm AI Hub CLI:\n"
                f"     See instructions in 'docs/qualcomm_ai_hub_workflow.md'.\n"
                f"  3. Place the exported .onnx or .bin file at: {self.path}\n"
                f"  (Or run with --mock or EXECUTION_PROVIDER=cpu to test logic in dev mode).\n"
            )
        else:
            msg += (
                f"  1. Place the CPU-compatible ONNX model at: {self.path}\n"
                f"  2. Or run the app in Mock / Dev mode using: python main.py --mock\n"
                f"     See 'models/README.md' for model download and setup guides.\n"
            )
        if suggestions:
            msg += f"  Additional details: {suggestions}\n"
        super().__init__(msg)


@dataclass
class ProviderInfo:
    """Introspected hardware and execution provider information."""
    requested_backend: str
    active_backend: str
    active_provider: str
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    display_status: str = "Initializing..."
    device_platform: str = field(default_factory=lambda: f"{platform.system()} ({platform.machine()})")
    available_providers: List[str] = field(default_factory=list)
    qnn_backend_path: str = "QnnHtp.dll"

    @property
    def is_npu(self) -> bool:
        return "QNN" in self.active_provider or self.active_backend == "qnn"


class InferenceEngine:
    """
    Unified Inference Engine for ONNX Runtime across CPU and Snapdragon Hexagon NPU.
    
    Adheres to:
      1. Single config-driven EXECUTION_PROVIDER setting ('cpu', 'qnn', 'cuda')
      2. Hardware abstraction: models are loaded through this engine only
      3. Automatic runtime detection and graceful fallback
      4. Clear error messages when model files are missing per target
    """

    def __init__(
        self,
        execution_provider: Optional[str] = None,
        qnn_backend_path: str = "QnnHtp.dll",
        allow_fallback: bool = True,
    ):
        """
        Initialize the inference engine.
        
        Args:
            execution_provider: 'cpu', 'qnn', or 'cuda'.
                                If None, checks env var 'EXECUTION_PROVIDER' then defaults to 'cpu'.
            qnn_backend_path: DLL name/path for Qualcomm Hexagon Tensor Processor (HTP).
            allow_fallback: If True, falls back to CPU if requested QNN/CUDA is not available.
        """
        # Determine requested backend from param or environment
        env_provider = os.environ.get("EXECUTION_PROVIDER")
        raw_backend = (execution_provider or env_provider or "cpu").strip().lower()

        if raw_backend in ("qnn", "hexagon", "npu"):
            self.requested_backend = ExecutionBackend.QNN
        elif raw_backend in ("cuda", "gpu"):
            self.requested_backend = ExecutionBackend.CUDA
        else:
            self.requested_backend = ExecutionBackend.CPU

        self.qnn_backend_path = qnn_backend_path
        self.allow_fallback = allow_fallback

        # Introspect available ONNX Runtime providers
        self.available_providers = ort.get_available_providers() if ORT_AVAILABLE else []
        self.provider_info = self._resolve_active_provider()

    def _resolve_active_provider(self) -> ProviderInfo:
        """Determines the active provider based on hardware capabilities and fallback rules."""
        req = self.requested_backend
        available = self.available_providers

        # Default info structure
        info = ProviderInfo(
            requested_backend=req.value,
            active_backend=req.value,
            active_provider="CPUExecutionProvider",
            is_fallback=False,
            available_providers=available,
            qnn_backend_path=self.qnn_backend_path,
        )

        if not ORT_AVAILABLE:
            info.active_backend = ExecutionBackend.CPU.value
            info.active_provider = "MockProvider (onnxruntime not installed)"
            info.display_status = "Running on: Mock Runtime (No ONNX)"
            return info

        # 1. Qualcomm Hexagon NPU (QNN)
        if req == ExecutionBackend.QNN:
            if "QNNExecutionProvider" in available:
                info.active_backend = ExecutionBackend.QNN.value
                info.active_provider = "QNNExecutionProvider"
                info.display_status = "Running on: Snapdragon Hexagon NPU (QNN)"
            elif self.allow_fallback:
                info.active_backend = ExecutionBackend.CPU.value
                info.active_provider = "CPUExecutionProvider"
                info.is_fallback = True
                info.fallback_reason = "QNNExecutionProvider not present in this ONNX Runtime build (x86 host detected)"
                info.display_status = "Running on: CPU (dev mode - QNN fallback)"
            else:
                raise RuntimeError(
                    f"Requested execution provider 'qnn' (QNNExecutionProvider) is not available. "
                    f"Available providers: {available}. Set allow_fallback: true or EXECUTION_PROVIDER=cpu."
                )

        # 2. CUDA (GPU)
        elif req == ExecutionBackend.CUDA:
            if "CUDAExecutionProvider" in available:
                info.active_backend = ExecutionBackend.CUDA.value
                info.active_provider = "CUDAExecutionProvider"
                info.display_status = "Running on: CUDA GPU"
            elif self.allow_fallback:
                info.active_backend = ExecutionBackend.CPU.value
                info.active_provider = "CPUExecutionProvider"
                info.is_fallback = True
                info.fallback_reason = "CUDAExecutionProvider not available; falling back to CPU"
                info.display_status = "Running on: CPU (dev mode - CUDA fallback)"
            else:
                raise RuntimeError("CUDAExecutionProvider requested but not available.")

        # 3. CPU Default
        else:
            info.active_backend = ExecutionBackend.CPU.value
            info.active_provider = "CPUExecutionProvider"
            info.display_status = "Running on: CPU (dev mode)"

        return info

    def verify_model_artifact(
        self,
        model_path: Union[str, Path],
        model_type: str = "Model",
        target: Optional[str] = None
    ) -> Path:
        """
        Validates that a required model file exists.
        
        Raises:
            MissingModelArtifactError: If file does not exist, explaining target-specific resolution.
        """
        path = Path(model_path)
        resolved_target = target or self.provider_info.active_backend

        if not path.exists():
            raise MissingModelArtifactError(
                model_type=model_type,
                target=resolved_target,
                path=path,
            )
        return path

    def create_session(
        self,
        model_path: Union[str, Path],
        model_type: str = "model",
        session_options: Optional[Any] = None,
        extra_qnn_options: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Creates an ONNX Runtime InferenceSession configured with the active execution provider.
        
        Args:
            model_path: Path to the .onnx model or QNN context binary.
            model_type: Descriptive name (e.g. 'Whisper STT', 'Speaker Diarization').
            session_options: Optional ort.SessionOptions object.
            extra_qnn_options: Additional key-value pairs for QNN EP tuning.
        
        Returns:
            ort.InferenceSession
        """
        if not ORT_AVAILABLE:
            raise RuntimeError(
                "onnxruntime is not installed. Install via 'pip install -r requirements-dev.txt'."
            )

        # Ensure model file exists; raises clean MissingModelArtifactError if missing
        verified_path = self.verify_model_artifact(
            model_path,
            model_type=model_type,
            target=self.provider_info.active_backend
        )

        # Prepare default session options if not provided
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Build providers and provider options according to active backend
        providers: List[str] = []
        provider_options: List[Dict[str, Any]] = []

        if self.provider_info.active_provider == "QNNExecutionProvider":
            # ==============================================================================
            # Qualcomm Hexagon NPU Configuration (Copilot+ PC Snapdragon X Elite / Plus)
            # TODO [Qualcomm AI Hub / Snapdragon]:
            #   - Tune 'htp_performance_mode' ('burst' for low latency, 'balanced' for efficiency)
            #   - Set 'vtcm_mb' memory allocation if custom weight partitioning is used
            #   - Set 'backend_path' to custom QnnHtp.dll location if not in system PATH
            # ==============================================================================
            qnn_opts = {
                "backend_path": self.qnn_backend_path,
                "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1",
            }
            if extra_qnn_options:
                qnn_opts.update(extra_qnn_options)

            providers = ["QNNExecutionProvider", "CPUExecutionProvider"]
            provider_options = [qnn_opts, {}]

        elif self.provider_info.active_provider == "CUDAExecutionProvider":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]

        else:
            # CPUExecutionProvider
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        return ort.InferenceSession(
            str(verified_path),
            sess_options=session_options,
            providers=providers,
            provider_options=provider_options,
        )

    def get_provider_info(self) -> ProviderInfo:
        """Returns the introspected provider info for UI and status reporting."""
        return self.provider_info
