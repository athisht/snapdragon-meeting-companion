"""
Speaker Diarization Module using ONNX Runtime.
Attributes audio segments and transcripts to discrete speakers (Speaker 1, Speaker 2, etc.).
Designed for lightweight on-device embedding models running on CPU (dev) or Hexagon NPU (Snapdragon).
Zero PyTorch or CUDA dependencies in core app logic.
"""
from typing import Dict, List, Optional
import numpy as np

from ..engine.inference_engine import InferenceEngine, MissingModelArtifactError


class SpeakerDiarizer:
    """
    Identifies and separates speakers across transcribed utterances.
    
    ========================================================================================
    TODO [Qualcomm AI Hub Integration - Speaker Embedding for Snapdragon]:
      1. Model: ResNet34-Speaker, CAM++, or Sherpa-ONNX 3D-Speaker.
         Export via Qualcomm AI Hub CLI on Snapdragon ARM64:
         python -m qai_hub_models.models.resnet34_speaker.export \
             --device "Snapdragon X Elite CRD" \
             --target-runtime onnx \
             --output-dir models/qnn/diarization/
      
      2. Embedding Extraction on Hexagon NPU:
         Pass preprocessed speech chunk into ONNX Runtime session configured with QNN EP:
         embedding = npu_session.run(None, {"audio_input": preprocessed_chunk})[0]
         embedding = embedding / np.linalg.norm(embedding)
         
      3. Online Clustering & Centroid Tracking:
         Compute cosine similarity against known speaker centroids:
         similarity = np.dot(embedding, centroid)
         - If similarity > similarity_threshold: assign to that speaker and update centroid.
         - Otherwise: register a new speaker identity (up to max_speakers).
    ========================================================================================
    """

    def __init__(
        self,
        engine: InferenceEngine,
        model_path: str = "models/cpu/diarization/speaker_embedding.onnx",
        similarity_threshold: float = 0.72,
        max_speakers: int = 6,
        mock_mode: bool = False,
    ):
        self.engine = engine
        self.model_path = model_path
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.mock_mode = mock_mode

        self._session = None
        # Active speaker profile embeddings (speaker_id -> centroid vector)
        self.speaker_centroids: Dict[str, np.ndarray] = {}
        self._mock_speaker_cycle = [
            "Speaker 1 (Athish)",
            "Speaker 2 (Qualcomm Team)",
            "Speaker 3 (Lead Architect)",
        ]
        self._mock_turn = 0

    def load_model(self):
        """Loads the on-device speaker embedding model via InferenceEngine."""
        if self.mock_mode:
            print(f"[Diarizer] Running in MOCK MODE ({self.engine.get_provider_info().display_status}).")
            return

        active_target = self.engine.get_provider_info().active_backend
        print(f"[Diarizer] Loading speaker embedding model for target [{active_target}] from: {self.model_path}")

        # create_session verifies path and raises clean MissingModelArtifactError if missing
        self._session = self.engine.create_session(
            self.model_path,
            model_type="Speaker Diarization Embedding",
        )
        print(f"[Diarizer] Successfully loaded speaker model on: {self.engine.get_provider_info().active_provider}")

    def extract_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extracts a normalized speaker embedding vector from an audio chunk.
        
        Args:
            audio_chunk: 1D numpy float32 array at 16000Hz.
        Returns:
            Normalized 1D embedding vector (length 192).
        """
        if self._session is not None:
            try:
                # Prepare input tensor for ONNX session
                inp_name = self._session.get_inputs()[0].name
                # Reshape as expected by embedding model (batch, samples)
                inp_data = audio_chunk.astype(np.float32).reshape(1, -1)
                raw_emb = self._session.run(None, {inp_name: inp_data})[0].flatten()
                norm = np.linalg.norm(raw_emb)
                if norm > 1e-6:
                    return raw_emb / norm
            except Exception as e:
                print(f"[Diarizer] ONNX embedding extraction failed: {e}. Using acoustic feature fallback.")

        # Pure NumPy acoustic feature extractor (energy + spectral moments)
        # Avoids external heavy PyTorch or CUDA dependencies
        vec = np.zeros(192, dtype=np.float32)
        if len(audio_chunk) > 0:
            # Energy, zero-crossing rate, spectral centroid
            fft = np.abs(np.fft.rfft(audio_chunk[:1024]))
            freqs = np.fft.rfftfreq(min(1024, len(audio_chunk)), 1.0 / 16000.0)
            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
            zcr = float(np.mean(np.abs(np.diff(np.sign(audio_chunk)))))
            rms = float(np.sqrt(np.mean(audio_chunk**2)))

            vec[0] = rms
            vec[1] = zcr
            vec[2] = centroid / 8000.0
            # Populate remainder with frequency bins
            n_fft_bins = min(189, len(fft))
            vec[3:3 + n_fft_bins] = fft[:n_fft_bins] / (np.max(fft) + 1e-8)

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def identify_speaker(self, audio_chunk: np.ndarray) -> str:
        """
        Identifies speaker label for the given audio chunk using cosine similarity clustering.
        
        Args:
            audio_chunk: 1D numpy array of the speech segment.
        Returns:
            Speaker identifier string (e.g. 'Speaker 1', 'Speaker 2').
        """
        if self.mock_mode:
            speaker = self._mock_speaker_cycle[self._mock_turn % len(self._mock_speaker_cycle)]
            self._mock_turn += 1
            return speaker

        embedding = self.extract_embedding(audio_chunk)

        # First speaker registration
        if not self.speaker_centroids:
            speaker_id = f"Speaker 1"
            self.speaker_centroids[speaker_id] = embedding
            return speaker_id

        # Calculate cosine similarity against all known speaker centroids
        best_speaker = None
        best_similarity = -1.0

        for spk_id, centroid in self.speaker_centroids.items():
            sim = float(np.dot(embedding, centroid))
            if sim > best_similarity:
                best_similarity = sim
                best_speaker = spk_id

        # Match threshold check
        if best_similarity >= self.similarity_threshold:
            # Update centroid with exponential moving average
            self.speaker_centroids[best_speaker] = (
                0.85 * self.speaker_centroids[best_speaker] + 0.15 * embedding
            )
            # Re-normalize centroid
            c_norm = np.linalg.norm(self.speaker_centroids[best_speaker])
            if c_norm > 1e-6:
                self.speaker_centroids[best_speaker] /= c_norm
            return best_speaker

        # If no match and max_speakers not reached, register new speaker
        if len(self.speaker_centroids) < self.max_speakers:
            new_id = f"Speaker {len(self.speaker_centroids) + 1}"
            self.speaker_centroids[new_id] = embedding
            return new_id

        # Fallback to closest match
        return best_speaker or "Speaker 1"

    def reset(self):
        """Clears active speaker centroids for a new session."""
        self.speaker_centroids.clear()
        self._mock_turn = 0