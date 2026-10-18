# Model Artifacts Directory (Hardware Target Separated)

This repository enforces strict separation between application code and offline model binaries. Model files are organized into dedicated subdirectories per execution target so that weights compiled for the Snapdragon Hexagon NPU can be dropped in on ARM64 hardware without modifying any codebase logic.

## Directory Structure

```
models/
├── cpu/                              # x86 Dev PC / Fallback Target
│   ├── stt/
│   │   └── whisper_base_en.onnx      # Standard ONNX Whisper model
│   ├── diarization/
│   │   └── speaker_embedding.onnx    # Standard ONNX speaker embedding model
│   └── llm/
│       └── llama_3_2_1b/             # ONNX / onnxruntime-genai model folder
│           ├── model.onnx
│           ├── genai_config.json
│           └── tokenizer.json
│
└── qnn/                              # Snapdragon Copilot+ PC Target (Hexagon NPU)
    ├── stt/
    │   └── whisper_base_en_quantized_qnn.onnx # Compiled via Qualcomm AI Hub
    ├── diarization/
    │   └── speaker_embedding_qnn.onnx         # Compiled via Qualcomm AI Hub
    └── llm/
        └── llama_3_2_1b_qnn/                  # ONNX Runtime GenAI with QNN EP
            ├── model.onnx
            ├── genai_config.json
            └── tokenizer.json
```

---

## Target-Specific Model Acquisition

### 1. For x86 Dev Machine (CPU Execution Provider)
To run real ONNX models locally on your x86 laptop:
- **Speech-to-Text**: Download or export any standard ONNX-format Whisper model into `models/cpu/stt/whisper_base_en.onnx`.
- **Diarization**: Place a ResNet34-Speaker or 3D-Speaker ONNX file into `models/cpu/diarization/speaker_embedding.onnx`.
- **LLM**: Place an ONNX Runtime GenAI compatible Llama-3.2-1B-Instruct or Phi-3.5-mini model folder into `models/cpu/llm/llama_3_2_1b/`.

> **Tip**: If you do not have model weights downloaded on your dev machine, run with `--mock` (`python main.py --mock`) or launch the app; if weights are missing, the pipeline will notify you and safely fall back to simulated audio/turns for UI and plumbing validation.

---

### 2. For Snapdragon Copilot+ PC (Qualcomm Hexagon NPU)
On your Snapdragon ARM64 laptop (Snapdragon X Elite / Plus):
1. Install Qualcomm AI Hub:
   ```bash
   pip install qai-hub
   qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_TOKEN>
   ```

2. Export models compiled specifically for the Hexagon Tensor Processor (HTP):
   ```bash
   # Whisper Base STT:
   python -m qai_hub_models.models.whisper_base_en.export \
       --device "Snapdragon X Elite CRD" \
       --target-runtime onnx \
       --output-dir models/qnn/stt/

   # Speaker Embedding:
   python -m qai_hub_models.models.resnet34_speaker.export \
       --device "Snapdragon X Elite CRD" \
       --target-runtime onnx \
       --output-dir models/qnn/diarization/

   # Llama 3.2 1B Instruct:
   python -m qai_hub_models.models.llama_v3_2_1b_instruct.export \
       --device "Snapdragon X Elite CRD" \
       --target-runtime onnx \
       --output-dir models/qnn/llm/llama_3_2_1b_qnn/
   ```

3. Set `EXECUTION_PROVIDER=qnn` in `.env` or run:
   ```bash
   python main.py --provider qnn
   ```
   The `InferenceEngine` will automatically bind to `models/qnn/` and activate `QNNExecutionProvider` on the Hexagon NPU.