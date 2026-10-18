# Qualcomm AI Hub Model Export & Snapdragon NPU Deployment Guide

This guide details how to export, compile, and drop in Qualcomm AI Hub models for **100% offline Hexagon NPU acceleration** on Snapdragon-powered HP Copilot+ PCs (ARM64 Windows).

---

## 🏗️ Hardware Architecture & Portability

The **Snapdragon Offline Meeting Companion** uses an **Inference Engine Hardware Abstraction Layer** built on top of ONNX Runtime. This means:
- The exact same Python pipeline code runs on your **x86 dev machine** (using `CPUExecutionProvider`) and on the **Snapdragon ARM64 target** (using `QNNExecutionProvider`).
- Only two things change when switching to Snapdragon hardware:
  1. The execution provider setting (`EXECUTION_PROVIDER=qnn`).
  2. The compiled model binaries located in `models/qnn/`.

```
                ┌──────────────────────────────────┐
                │   InferenceEngine Abstraction    │
                └──────────────┬───────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [x86 Dev Machine]             [Snapdragon Copilot+ PC]
    CPUExecutionProvider               QNNExecutionProvider
     models/cpu/*.onnx                  models/qnn/*.onnx / .bin
   "Running on: CPU (dev)"            "Running on: Snapdragon NPU"
```

---

## 🛠️ Step 1: Prerequisites on Snapdragon ARM64 PC

When you clone the repository onto your Snapdragon-powered HP laptop:

1. **Ensure Native ARM64 Python is installed**:
   Install Python 3.10+ (ARM64 Windows build from python.org or Windows Store).

2. **Install ARM64 Dependencies**:
   ```powershell
   pip install -r requirements-arm.txt
   ```

3. **Install Qualcomm AI Hub Client**:
   ```powershell
   pip install qai-hub
   ```

4. **Authenticate with Qualcomm AI Hub**:
   Sign up or log in at [Qualcomm AI Hub](https://aihub.qualcomm.com/) and grab your API token:
   ```powershell
   qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_API_TOKEN>
   ```

---

## 📦 Step 2: Export & Compile Models for Hexagon NPU

Qualcomm AI Hub compiles models specifically optimized for the Hexagon Tensor Processor (HTP) on Snapdragon X Elite and Snapdragon X Plus silicon.

### 1. Whisper Base Speech-to-Text
Export the quantized Whisper speech-to-text model:
```powershell
python -m qai_hub_models.models.whisper_base_en.export `
    --device "Snapdragon X Elite CRD" `
    --target-runtime onnx `
    --output-dir models/qnn/stt/
```
*Destination path:* `models/qnn/stt/whisper_base_en_quantized_qnn.onnx`

### 2. Speaker Diarization / Voice Embedding
Export an on-device speaker embedding extractor:
```powershell
python -m qai_hub_models.models.resnet34_speaker.export `
    --device "Snapdragon X Elite CRD" `
    --target-runtime onnx `
    --output-dir models/qnn/diarization/
```
*Destination path:* `models/qnn/diarization/speaker_embedding_qnn.onnx`

### 3. Local LLM (Llama 3.2 1B or Phi-3.5-mini)
Export Llama 3.2 1B Instruct for Hexagon NPU inference via ONNX Runtime GenAI:
```powershell
python -m qai_hub_models.models.llama_v3_2_1b_instruct.export `
    --device "Snapdragon X Elite CRD" `
    --target-runtime onnx `
    --output-dir models/qnn/llm/llama_3_2_1b_qnn/
```
*Destination path:* `models/qnn/llm/llama_3_2_1b_qnn/` (contains `model.onnx`, `genai_config.json`, and tokenizer files)

---

## ⚙️ Step 3: Activate Snapdragon Hexagon NPU

Once the compiled model files are placed in `models/qnn/`, activate the QNN execution provider using any of the following methods:

### Method A: Environment Variable (Recommended for Demos)
In PowerShell:
```powershell
$env:EXECUTION_PROVIDER="qnn"
python main.py
```
Or set in `.env`:
```ini
EXECUTION_PROVIDER=qnn
```

### Method B: Command Line Flag
```powershell
python main.py --provider qnn
```

### Method C: `config/settings.yaml`
```yaml
device:
  execution_provider: "qnn"
  qnn_backend_path: "QnnHtp.dll"
```

---

## 🔍 Step 4: Verification & Demo Confirmation

When you launch the app on the Snapdragon laptop:

1. **Check the Top Header Badge**:
   The status badge will display in vibrant green:
   ```
   ⚡ Running on: Snapdragon Hexagon NPU (QNN)
   ```
2. **Check Windows Task Manager**:
   Open Windows Task Manager, switch to the **Performance** tab, and observe the **NPU** graph. During speech transcription and summary updates, compute spikes will reflect directly on the Hexagon NPU with 0% cloud network traffic.

3. **Check Hardware Info Dialog**:
   Click **⚙ Hardware Info** in the app header to view active ONNX providers, HTP DLL paths, and model directory resolutions.

---

## 🛡️ Troubleshooting & Fallback Behavior

- **"QNNExecutionProvider not present in this ONNX Runtime build"**:
  Ensure you installed `onnxruntime-qnn` from `requirements-arm.txt`. If run on an x86 machine or if QNN is missing, the `InferenceEngine` will gracefully fall back to CPU dev mode (`Running on: CPU (dev mode - QNN fallback)`).
- **QnnHtp.dll location**:
  The Qualcomm Hexagon HTP library is included in the Snapdragon Windows on ARM OS image and the Qualcomm AI Engine Direct SDK. If not in the system PATH, specify the absolute path in `config/settings.yaml` under `device.qnn_backend_path`.
