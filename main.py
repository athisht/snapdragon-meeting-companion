"""
Snapdragon Offline Meeting Companion - Entry Point
Runs 100% locally on Snapdragon Copilot+ PCs (Hexagon NPU) and portable to x86 Dev PCs (CPU).
Zero cloud dependencies.
"""
import sys
import argparse
from pathlib import Path

from src.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline Meeting Companion for Snapdragon Copilot+ PCs and x86 Dev Machines"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration YAML file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["cpu", "qnn", "cuda"],
        default=None,
        help="Override execution provider backend ('cpu' for x86 dev, 'qnn' for Snapdragon Hexagon NPU)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode (simulated pipeline for UI and plumbing validation)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Force real ONNX inference mode (requires models in models/cpu/ or models/qnn/)",
    )
    parser.add_argument(
        "--cli-test",
        action="store_true",
        help="Run headless pipeline smoke test in terminal without launching UI",
    )
    return parser.parse_args()


def run_cli_test(config: Config):
    """Headless verification test for the audio->STT->diarizer->LLM pipeline."""
    import time
    from src.pipeline import MeetingPipeline

    print("\n" + "=" * 70)
    print("  SNAPDRAGON OFFLINE MEETING COMPANION - PIPELINE SMOKE TEST")
    print("=" * 70)

    pipeline = MeetingPipeline(config)
    p = pipeline.provider_info
    print(f"Platform:           {p.device_platform}")
    print(f"Requested Backend:  {p.requested_backend}")
    print(f"Active Provider:    {p.active_provider}")
    print(f"Display Status:     {p.display_status}")
    if p.is_fallback:
        print(f"Fallback Notice:    {p.fallback_reason}")
    print(f"Mock Mode:          {config.app.mock_mode}")

    def on_transcript(entry):
        print(f"\n[TRANSCRIPT] [{entry['timestamp']}] {entry['speaker']}:")
        print(f"  \"{entry['text']}\"")

    def on_summary(summary):
        print("\n" + "-" * 50)
        print("[AI MEETING SUMMARY UPDATED]")
        for bp in summary.get("bullet_points", []):
            print(f"  * {bp}")
        print("[ACTION ITEMS]")
        for act in summary.get("action_items", []):
            print(f"  [ ] [{act['owner']}] {act['task']} ({act['priority']})")
        print("-" * 50)

    def on_status(status):
        print(f"[PIPELINE STATUS] {status}")

    def on_error(err):
        print(f"[PIPELINE NOTIFICATION]\n{err}")

    pipeline.signals.transcript_added.connect(on_transcript)
    pipeline.signals.summary_updated.connect(on_summary)
    pipeline.signals.status_changed.connect(on_status)
    pipeline.signals.error_occurred.connect(on_error)

    pipeline.initialize_models()
    pipeline.start()

    print("\n[Test] Pipeline running for 12 seconds to simulate meeting turns...")
    try:
        time.sleep(12)
    except KeyboardInterrupt:
        pass

    pipeline.stop()
    export_file = "exports/test_cli_meeting_notes.md"
    saved = pipeline.export_notes(export_file, fmt="md")
    print(f"\n[Test Completed] Meeting notes exported to: {saved}\n")


def main():
    args = parse_args()
    config = Config.load(args.config)

    # CLI flag overrides
    if args.provider:
        config.device.execution_provider = args.provider
    if args.mock:
        config.app.mock_mode = True
    elif args.real:
        config.app.mock_mode = False

    if args.cli_test:
        run_cli_test(config)
        return

    # Desktop GUI Launch
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(config.app.name)
    app.setOrganizationName("Snapdragon Hackathon")

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()