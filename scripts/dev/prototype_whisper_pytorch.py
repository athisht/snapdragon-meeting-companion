"""
DEV-ONLY SCRIPT: PyTorch Prototype & ONNX Export Helper.
========================================================================================
IMPORTANT NOTICE:
This script is for local x86 development, testing, and ONNX conversion ONLY.
It is NOT part of the shipped meeting companion application.
The shipped app relies strictly on the hardware-portable ONNX Runtime InferenceEngine,
ensuring 100% compatibility across both x86 development PCs and Snapdragon ARM64 laptops.
========================================================================================
"""
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Dev-Only PyTorch prototype and ONNX conversion tool"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small"],
        help="Whisper model size to export to ONNX format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/cpu/stt",
        help="Directory to save the exported ONNX model",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  DEV-ONLY: PyTorch Whisper ONNX Export Tool")
    print("=" * 70)
    print(f"Target size: {args.model_size}")
    print(f"Output path: {args.output_dir}\n")

    try:
        import torch
        print(f"[OK] PyTorch version detected: {torch.__version__}")
        print(f"[OK] CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("[Note] PyTorch is not installed in this environment.")
        print("To run PyTorch prototype conversions on your x86 dev machine:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nNote: The core application does NOT require PyTorch and runs purely on ONNX Runtime.")
        return

    print("\nTo export Whisper to ONNX using PyTorch:")
    print("  1. Load PyTorch Whisper model: model = whisper.load_model('base')")
    print("  2. Export encoder/decoder using torch.onnx.export(...)")
    print("  3. Save exported model into: models/cpu/stt/whisper_base_en.onnx")
    print("\nFor Snapdragon Copilot+ PC deployment, prefer using Qualcomm AI Hub:")
    print("  See 'docs/qualcomm_ai_hub_workflow.md' for the recommended QNN export workflow.")


if __name__ == "__main__":
    main()
