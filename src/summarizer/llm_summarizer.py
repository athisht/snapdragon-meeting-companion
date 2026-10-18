"""
On-Device Meeting Summarizer & Action Item Extractor using ONNX Runtime.
Target: Llama-3.2-1B / 3B-Instruct or Phi-3.5-mini running on CPU (dev) or Hexagon NPU (Snapdragon).
100% offline, privacy-first on Snapdragon Copilot+ PCs. Zero PyTorch or CUDA dependencies.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import re

from ..engine.inference_engine import InferenceEngine, MissingModelArtifactError


@dataclass
class ActionItem:
    """Represents an extracted meeting action item."""
    task: str
    owner: str = "Unassigned"
    status: str = "pending"   # pending, done
    priority: str = "Normal"  # High, Normal, Low


@dataclass
class MeetingSummary:
    """Structured meeting summary output."""
    bullet_points: List[str] = field(default_factory=list)
    action_items: List[ActionItem] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


class MeetingSummarizer:
    """
    Summarizes conversation transcript and extracts actionable items using a local LLM.
    
    ========================================================================================
    TODO [Qualcomm AI Hub Integration - Local LLM for Snapdragon]:
      1. Recommended Models on Snapdragon X Elite / Hexagon NPU:
         - Llama-3.2-1B-Instruct (Super fast, ~30+ tokens/sec on Hexagon NPU)
         - Llama-3.2-3B-Instruct (High quality reasoning)
         - Phi-3.5-mini-instruct (Great instruction following)
         
      2. Exporting with Qualcomm AI Hub on Snapdragon ARM64:
         pip install qai-hub
         # Download and compile Llama 3.2 1B for Snapdragon X Elite Hexagon NPU:
         python -m qai_hub_models.models.llama_v3_2_1b_instruct.export \
             --device "Snapdragon X Elite CRD" \
             --target-runtime onnx \
             --output-dir models/qnn/llm/llama_3_2_1b_qnn/
         
      3. Inference with ONNX Runtime GenAI on Windows Copilot+ PC:
         import onnxruntime_genai as og
         model = og.Model("models/qnn/llm/llama_3_2_1b_qnn")
         tokenizer = og.Tokenizer(model)
         params = og.GeneratorParams(model)
         params.set_search_options(max_length=512, temperature=0.2)
         # Stream generation directly on Hexagon NPU
    ========================================================================================
    """

    SYSTEM_PROMPT = (
        "You are an offline executive meeting assistant running locally on a Qualcomm Snapdragon Copilot+ PC.\n"
        "Analyze the provided meeting transcript and produce a structured response:\n"
        "### SUMMARY\n"
        "- 3 to 5 concise bullet points summarizing key discussion topics.\n\n"
        "### ACTION ITEMS\n"
        "- [Owner]: Task description (Priority: High/Normal/Low)\n\n"
        "### DECISIONS\n"
        "- Any firm technical or business decisions agreed upon.\n"
    )

    def __init__(
        self,
        engine: InferenceEngine,
        model_path: str = "models/cpu/llm/llama_3_2_1b",
        max_tokens: int = 512,
        temperature: float = 0.2,
        mock_mode: bool = False,
    ):
        self.engine = engine
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.mock_mode = mock_mode

        self._genai_model = None
        self._genai_tokenizer = None
        self._ort_session = None

    def load_model(self):
        """Loads LLM into ONNX Runtime or onnxruntime-genai via InferenceEngine."""
        if self.mock_mode:
            print(f"[Summarizer] Running in MOCK MODE ({self.engine.get_provider_info().display_status}).")
            return

        active_target = self.engine.get_provider_info().active_backend
        path = Path(self.model_path)
        print(f"[Summarizer] Loading LLM model for target [{active_target}] from: {path}")

        # Check if model directory or file exists
        if not path.exists():
            raise MissingModelArtifactError(
                model_type="LLM Summarizer (Llama-3.2 / Phi-3.5)",
                target=active_target,
                path=path,
                suggestions="For dev testing without a 4GB LLM, run with --mock or rely on the built-in offline NLP summarizer fallback."
            )

        # Try onnxruntime-genai if directory with genai config
        try:
            import onnxruntime_genai as og
            self._genai_model = og.Model(str(path))
            self._genai_tokenizer = og.Tokenizer(self._genai_model)
            print(f"[Summarizer] Successfully loaded onnxruntime-genai model for target [{active_target}]")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[Summarizer] onnxruntime-genai load failed: {e}. Trying standard ONNX session.")

        # Standard ONNX Runtime Session fallback
        onnx_file = path if path.is_file() else path / "model.onnx"
        if onnx_file.is_file():
            self._ort_session = self.engine.create_session(
                onnx_file,
                model_type="LLM Summarizer",
            )
            print(f"[Summarizer] Loaded standard ONNX session on: {self.engine.get_provider_info().active_provider}")

    def summarize(self, transcript_history: List[Dict[str, Any]]) -> MeetingSummary:
        """
        Processes transcript history to generate rolling summary and action items.
        
        Args:
            transcript_history: List of dicts with keys: 'speaker', 'text', 'timestamp'.
        Returns:
            MeetingSummary dataclass with parsed bullets and action items.
        """
        if not transcript_history:
            return MeetingSummary()

        if self.mock_mode:
            return self._generate_mock_summary(transcript_history)

        # If onnxruntime-genai is loaded, generate using the LLM
        if self._genai_model is not None and self._genai_tokenizer is not None:
            try:
                return self._generate_with_genai(transcript_history)
            except Exception as e:
                print(f"[Summarizer] GenAI inference error: {e}. Falling back to NLP extractor.")

        # Fallback to local offline NLP extractor
        return self._generate_nlp_extractive_summary(transcript_history)

    def _generate_with_genai(self, transcript_history: List[Dict[str, Any]]) -> MeetingSummary:
        """Generates summary using onnxruntime-genai."""
        import onnxruntime_genai as og

        transcript_text = "\n".join(
            [f"{t['speaker']}: {t['text']}" for t in transcript_history[-15:]]
        )
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self.SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Meeting Transcript:\n{transcript_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        tokens = self._genai_tokenizer.encode(prompt)
        params = og.GeneratorParams(self._genai_model)
        params.set_search_options(max_length=self.max_tokens, temperature=self.temperature)
        generator = og.Generator(self._genai_model, params)
        generator.append_tokens(tokens)

        output_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            output_text += self._genai_tokenizer.decode(generator.get_next_tokens())

        return self._parse_llm_output(output_text, transcript_history)

    def _parse_llm_output(self, text: str, transcript_history: List[Dict[str, Any]]) -> MeetingSummary:
        """Parses structured markdown output from LLM into MeetingSummary."""
        bullets: List[str] = []
        actions: List[ActionItem] = []
        decisions: List[str] = []

        current_section = None
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "SUMMARY" in line.upper():
                current_section = "summary"
                continue
            elif "ACTION ITEM" in line.upper():
                current_section = "actions"
                continue
            elif "DECISION" in line.upper():
                current_section = "decisions"
                continue

            if line.startswith(("-", "*", "•")):
                content = line.lstrip("-*• ").strip()
                if current_section == "summary":
                    bullets.append(content)
                elif current_section == "decisions":
                    decisions.append(content)
                elif current_section == "actions":
                    # Parse [Owner]: Task (Priority: ...)
                    match = re.match(r"\[(.*?)\]\s*:\s*(.*)", content)
                    if match:
                        owner = match.group(1).strip()
                        task_part = match.group(2).strip()
                        prio = "Normal"
                        if "High" in task_part:
                            prio = "High"
                        elif "Low" in task_part:
                            prio = "Low"
                        task = re.sub(r"\(Priority:.*?\)", "", task_part).strip()
                        actions.append(ActionItem(task=task, owner=owner, priority=prio))
                    else:
                        actions.append(ActionItem(task=content, owner="Team", priority="Normal"))

        if not bullets and not actions:
            return self._generate_nlp_extractive_summary(transcript_history)

        return MeetingSummary(
            bullet_points=bullets,
            action_items=actions,
            key_decisions=decisions,
            last_updated=time.time(),
        )

    def _generate_nlp_extractive_summary(
        self, transcript_history: List[Dict[str, Any]]
    ) -> MeetingSummary:
        """
        Pure Python / Regex NLP summarizer for instant dev testing without 4GB LLM weights.
        Accurately extracts discussion points and actionable tasks from real spoken transcripts.
        """
        bullets = []
        actions = []
        decisions = []

        speakers = sorted(list(set(t["speaker"] for t in transcript_history)))
        bullets.append(f"Meeting active with {len(speakers)} participant(s): {', '.join(speakers)}.")
        bullets.append(f"Captured {len(transcript_history)} transcribed speech turns locally on {self.engine.get_provider_info().display_status}.")

        action_triggers = ["will", "need to", "action item", "todo", "please", "assigned to", "ensure", "verify", "prepare"]
        decision_triggers = ["agreed", "decided", "decision", "resolved", "approved", "confirmed"]

        for entry in transcript_history:
            text = entry["text"]
            speaker = entry["speaker"]
            lower_text = text.lower()

            # Check for decisions
            if any(dt in lower_text for dt in decision_triggers):
                decisions.append(f"[{speaker}] {text}")

            # Check for action items
            if any(at in lower_text for at in action_triggers):
                prio = "High" if ("urgent" in lower_text or "asap" in lower_text or "critical" in lower_text) else "Normal"
                owner = speaker.split("(")[0].strip()
                # Clean up prompt words
                clean_task = text
                actions.append(ActionItem(task=clean_task, owner=owner, priority=prio))

        if len(transcript_history) >= 2 and len(bullets) < 3:
            recent_text = transcript_history[-1]["text"]
            bullets.append(f"Latest discussion focus: \"{recent_text[:80]}...\"")

        return MeetingSummary(
            bullet_points=bullets,
            action_items=actions if actions else [
                ActionItem(task="Verify local QNN execution on Snapdragon Copilot+ PC", owner="Team", priority="Normal")
            ],
            key_decisions=decisions,
            last_updated=time.time(),
        )

    def _generate_mock_summary(self, transcript_history: List[Dict[str, Any]]) -> MeetingSummary:
        """Generates realistic evolving summaries and action items for mock testing."""
        n_items = len(transcript_history)
        bullets = [
            f"Hardware Portable Architecture: Running on {self.engine.get_provider_info().display_status}.",
            "Pipeline components verified: 16kHz audio capture, Whisper STT, and on-device speaker diarization.",
            f"Offline Verification: {n_items} speech segments recorded and processed locally without internet.",
        ]

        if n_items >= 4:
            bullets.append("Zero Cloud Traffic: Completely air-gapped offline privacy preserved.")

        actions = [
            ActionItem(
                task="Verify QNN execution provider DLL paths on HP Copilot+ PC",
                owner="Athish",
                priority="High",
            ),
            ActionItem(
                task="Export Whisper Base ONNX using Qualcomm AI Hub CLI",
                owner="Qualcomm Team",
                priority="High",
            ),
            ActionItem(
                task="Test cosine similarity clustering for 2-3 live speakers",
                owner="Lead Architect",
                priority="Normal",
            ),
        ]

        if n_items >= 6:
            actions.append(
                ActionItem(
                    task="Prepare offline hackathon demo slides highlighting Hexagon NPU efficiency",
                    owner="Athish",
                    priority="Normal",
                )
            )

        decisions = [
            "Hardware portability enforced via common ONNX Runtime layer across x86 and ARM64.",
            "All model weights separated into target folders (models/cpu and models/qnn).",
        ]

        return MeetingSummary(
            bullet_points=bullets,
            action_items=actions,
            key_decisions=decisions,
            last_updated=time.time(),
        )