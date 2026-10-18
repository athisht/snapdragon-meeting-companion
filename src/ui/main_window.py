"""
Main Desktop Application Window
Built with PySide6 (Qt for Python) for snappy, native performance on Windows 11 Copilot+ PCs.
Displays prominent hardware execution provider status (CPU dev mode vs Snapdragon Hexagon NPU).
"""
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon, QFont, QColor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QSplitter, QScrollArea, QFrame,
    QCheckBox, QMessageBox, QFileDialog, QDialog, QTextBrowser,
    QGridLayout
)

from ..config import Config
from ..pipeline import MeetingPipeline
from ..engine.inference_engine import ProviderInfo
from .styles import DARK_THEME_QSS, SPEAKER_COLORS


class UtteranceCard(QFrame):
    """Visual card for a single transcribed utterance with speaker attribution."""
    def __init__(self, utterance: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Header: Speaker Badge + Timestamp
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        speaker_name = utterance.get("speaker", "Unknown Speaker")
        timestamp_str = utterance.get("timestamp", "--:--:--")

        # Color scheme based on speaker
        spk_idx = 0
        if "Speaker 2" in speaker_name:
            spk_idx = 1
        elif "Speaker 3" in speaker_name:
            spk_idx = 2
        elif "Speaker 4" in speaker_name:
            spk_idx = 3
        elif "Speaker 5" in speaker_name:
            spk_idx = 4
        color = SPEAKER_COLORS[spk_idx % len(SPEAKER_COLORS)]

        speaker_badge = QLabel(f" {speaker_name} ")
        speaker_badge.setStyleSheet(
            f"background-color: {color['bg']}; color: {color['text']}; "
            f"border: 1px solid {color['border']}; border-radius: 10px; "
            f"padding: 2px 8px; font-weight: bold; font-size: 11px;"
        )

        time_label = QLabel(timestamp_str)
        time_label.setStyleSheet("color: #8b949e; font-size: 11px; font-family: monospace;")

        conf = utterance.get("confidence", 1.0)
        conf_badge = QLabel(f"Conf: {int(conf*100)}%")
        conf_badge.setStyleSheet("color: #3fb950; font-size: 10px;")

        header_layout.addWidget(speaker_badge)
        header_layout.addWidget(time_label)
        header_layout.addStretch()
        header_layout.addWidget(conf_badge)

        # Utterance Text
        text_label = QLabel(utterance.get("text", ""))
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #e6edf3; font-size: 13px; line-height: 1.4;")
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addLayout(header_layout)
        layout.addWidget(text_label)


class ActionItemRow(QFrame):
    """Interactive checklist item for extracted action items."""
    def __init__(self, action: Dict[str, Any], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(action.get("status") == "done")
        self.checkbox.stateChanged.connect(self._on_checked)

        owner = action.get("owner", "Unassigned")
        owner_badge = QLabel(f"@{owner}")
        owner_badge.setStyleSheet(
            "background-color: #1f2937; color: #60a5fa; border: 1px solid #3b82f6; "
            "border-radius: 6px; padding: 2px 6px; font-size: 10px; font-weight: bold;"
        )

        priority = action.get("priority", "Normal")
        prio_color = "#f85149" if priority.lower() == "high" else "#8b949e"
        prio_badge = QLabel(f"[{priority}]")
        prio_badge.setStyleSheet(f"color: {prio_color}; font-size: 10px; font-weight: bold;")

        self.task_label = QLabel(action.get("task", ""))
        self.task_label.setWordWrap(True)
        self.task_label.setStyleSheet("color: #e6edf3; font-size: 12px;")

        layout.addWidget(self.checkbox)
        layout.addWidget(owner_badge)
        layout.addWidget(self.task_label, stretch=1)
        layout.addWidget(prio_badge)

        self.setStyleSheet("background-color: #21262d; border-radius: 6px;")

    def _on_checked(self, state):
        if state == Qt.CheckState.Checked.value:
            self.task_label.setStyleSheet("color: #6e7681; text-decoration: line-through; font-size: 12px;")
        else:
            self.task_label.setStyleSheet("color: #e6edf3; font-size: 12px;")


class MainWindow(QMainWindow):
    """Main desktop application window for Snapdragon Offline Meeting Companion."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.pipeline = MeetingPipeline(config)

        self.meeting_seconds = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_timer)

        self._init_ui()
        self._connect_signals()

        # Update initial hardware status
        self._update_hardware_badge(self.pipeline.provider_info)

        # Initialize models (or mock models)
        self.pipeline.initialize_models()

    def _init_ui(self):
        self.setWindowTitle(self.config.ui.window_title)
        self.resize(self.config.ui.width, self.config.ui.height)
        self.setStyleSheet(DARK_THEME_QSS)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # 1. Top Header Card
        main_layout.addWidget(self._create_header())

        # 2. Main Workspace (Splitter: Left Transcript | Right Intelligence)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_transcript_panel())
        splitter.addWidget(self._create_intelligence_panel())
        splitter.setStretchFactor(0, 3) # 60% Transcript
        splitter.setStretchFactor(1, 2) # 40% Summary/Actions
        main_layout.addWidget(splitter, stretch=1)

        # 3. Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Ready. Running on: {self.pipeline.provider_info.display_status}")

    def _create_header(self) -> QWidget:
        header_frame = QFrame()
        header_frame.setObjectName("headerCard")
        header_frame.setProperty("class", "headerCard")
        layout = QHBoxLayout(header_frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(14)

        # App branding & Hardware Execution Status Badge
        branding_layout = QVBoxLayout()
        branding_layout.setSpacing(4)

        app_title = QLabel("⚡ Snapdragon Meeting Companion")
        app_title.setProperty("class", "titleLabel")

        badges_layout = QHBoxLayout()
        badges_layout.setSpacing(8)

        # Prominent Hardware Execution Badge
        self.hardware_badge = QLabel("Detecting Hardware...")
        self.hardware_badge.setProperty("class", "badgeCPU")
        badges_layout.addWidget(self.hardware_badge)

        if self.config.app.mock_mode:
            mock_badge = QLabel("MOCK SKELETON MODE")
            mock_badge.setProperty("class", "badgeMock")
            badges_layout.addWidget(mock_badge)

        badges_layout.addStretch()

        branding_layout.addWidget(app_title)
        branding_layout.addLayout(badges_layout)
        layout.addLayout(branding_layout)

        layout.addStretch()

        # Audio VU Level Meter
        audio_layout = QVBoxLayout()
        audio_layout.setSpacing(3)
        audio_label = QLabel("MIC INPUT LEVEL")
        audio_label.setStyleSheet("font-size: 10px; color: #8b949e; font-weight: 600;")
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setFixedWidth(130)
        self.vu_meter.setTextVisible(False)
        audio_layout.addWidget(audio_label)
        audio_layout.addWidget(self.vu_meter)
        layout.addLayout(audio_layout)

        # Meeting Timer
        timer_layout = QVBoxLayout()
        timer_layout.setSpacing(3)
        timer_label = QLabel("SESSION TIME")
        timer_label.setStyleSheet("font-size: 10px; color: #8b949e; font-weight: 600;")
        self.timer_display = QLabel("00:00:00")
        self.timer_display.setProperty("class", "timerLabel")
        timer_layout.addWidget(timer_label)
        timer_layout.addWidget(self.timer_display)
        layout.addLayout(timer_layout)

        # Primary Controls
        self.btn_start = QPushButton("▶ Start Meeting")
        self.btn_start.setProperty("class", "accentButton")
        self.btn_start.clicked.connect(self._toggle_meeting)

        self.btn_clear = QPushButton("🧹 Clear")
        self.btn_clear.clicked.connect(self._clear_session)

        self.btn_export = QPushButton("💾 Export Notes")
        self.btn_export.clicked.connect(self._export_dialog)

        self.btn_config = QPushButton("⚙ Hardware Info")
        self.btn_config.clicked.connect(self._show_config_dialog)

        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_clear)
        layout.addWidget(self.btn_export)
        layout.addWidget(self.btn_config)

        return header_frame

    def _update_hardware_badge(self, info: ProviderInfo):
        """Updates prominent execution provider status in the header."""
        if info.is_npu:
            self.hardware_badge.setText(f"⚡ {info.display_status}")
            self.hardware_badge.setProperty("class", "badgeNPU")
            self.hardware_badge.setToolTip("Qualcomm Hexagon NPU active via QNNExecutionProvider.")
        elif info.is_fallback:
            self.hardware_badge.setText(f"⚠ {info.display_status}")
            self.hardware_badge.setProperty("class", "badgeFallback")
            self.hardware_badge.setToolTip(f"Fallback reason: {info.fallback_reason}")
        else:
            self.hardware_badge.setText(f"🖥 {info.display_status}")
            self.hardware_badge.setProperty("class", "badgeCPU")
            self.hardware_badge.setToolTip("Running on CPU via CPUExecutionProvider (x86 development mode).")

        self.hardware_badge.setStyle(self.hardware_badge.style())

    def _create_transcript_panel(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("class", "card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("Live Transcript")
        title.setProperty("class", "sectionHeader")
        self.utterance_count_lbl = QLabel("0 utterances")
        self.utterance_count_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.utterance_count_lbl)
        layout.addLayout(hdr)

        # Scroll area for utterances
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: transparent;")

        self.transcript_container = QWidget()
        self.transcript_layout = QVBoxLayout(self.transcript_container)
        self.transcript_layout.setContentsMargins(0, 0, 0, 0)
        self.transcript_layout.setSpacing(8)
        self.transcript_layout.addStretch()

        self.scroll_area.setWidget(self.transcript_container)
        layout.addWidget(self.scroll_area)

        return panel

    def _create_intelligence_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Card 1: Rolling Summary
        summary_card = QFrame()
        summary_card.setProperty("class", "card")
        sum_layout = QVBoxLayout(summary_card)
        sum_layout.setContentsMargins(12, 10, 12, 10)
        sum_layout.setSpacing(8)

        sum_hdr = QHBoxLayout()
        sum_title = QLabel("Rolling Executive Summary")
        sum_title.setProperty("class", "sectionHeader")
        self.summary_updated_lbl = QLabel("Awaiting input")
        self.summary_updated_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        sum_hdr.addWidget(sum_title)
        sum_hdr.addStretch()
        sum_hdr.addWidget(self.summary_updated_lbl)
        sum_layout.addLayout(sum_hdr)

        self.summary_browser = QTextBrowser()
        self.summary_browser.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; "
            "color: #e6edf3; padding: 10px; font-size: 13px; line-height: 1.5;"
        )
        self.summary_browser.setHtml(
            "<p style='color: #8b949e; font-style: italic;'>Meeting summary will generate dynamically as participants speak...</p>"
        )
        sum_layout.addWidget(self.summary_browser)
        layout.addWidget(summary_card, stretch=2)

        # Card 2: Action Items & Decisions
        actions_card = QFrame()
        actions_card.setProperty("class", "card")
        act_layout = QVBoxLayout(actions_card)
        act_layout.setContentsMargins(12, 10, 12, 10)
        act_layout.setSpacing(8)

        act_hdr = QHBoxLayout()
        act_title = QLabel("Extracted Action Items & Decisions")
        act_title.setProperty("class", "sectionHeader")
        self.actions_count_lbl = QLabel("0 items")
        self.actions_count_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        act_hdr.addWidget(act_title)
        act_hdr.addStretch()
        act_hdr.addWidget(self.actions_count_lbl)
        act_layout.addLayout(act_hdr)

        self.actions_scroll = QScrollArea()
        self.actions_scroll.setWidgetResizable(True)
        self.actions_scroll.setStyleSheet("border: none; background: transparent;")

        self.actions_container = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_container)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.actions_layout.setSpacing(6)
        self.actions_layout.addStretch()

        self.actions_scroll.setWidget(self.actions_container)
        act_layout.addWidget(self.actions_scroll)
        layout.addWidget(actions_card, stretch=3)

        return panel

    def _connect_signals(self):
        """Connects pipeline worker signals to UI update slots."""
        self.pipeline.signals.audio_level.connect(self._on_audio_level)
        self.pipeline.signals.transcript_added.connect(self._on_transcript_added)
        self.pipeline.signals.summary_updated.connect(self._on_summary_updated)
        self.pipeline.signals.status_changed.connect(self._on_status_changed)
        self.pipeline.signals.hardware_status_changed.connect(self._on_hardware_changed)
        self.pipeline.signals.error_occurred.connect(self._on_error)

    @Slot(object)
    def _on_hardware_changed(self, info: ProviderInfo):
        self._update_hardware_badge(info)

    @Slot(float)
    def _on_audio_level(self, level: float):
        percent = int(max(0.0, min(1.0, level)) * 100)
        self.vu_meter.setValue(percent)

    @Slot(dict)
    def _on_transcript_added(self, utterance: Dict[str, Any]):
        card = UtteranceCard(utterance, parent=self.transcript_container)
        count = self.transcript_layout.count()
        self.transcript_layout.insertWidget(max(0, count - 1), card)

        total = len(self.pipeline.transcript_history)
        self.utterance_count_lbl.setText(f"{total} utterances")
        QTimer.singleShot(50, self._scroll_transcript_to_bottom)

    def _scroll_transcript_to_bottom(self):
        bar = self.scroll_area.verticalScrollBar()
        bar.setValue(bar.maximum())

    @Slot(dict)
    def _on_summary_updated(self, summary_data: Dict[str, Any]):
        bullets = summary_data.get("bullet_points", [])
        decisions = summary_data.get("key_decisions", [])

        html_parts = []
        if bullets:
            html_parts.append("<b style='color: #58a6ff;'>Key Discussion Points:</b><ul style='margin-top: 4px;'>")
            for b in bullets:
                html_parts.append(f"<li style='margin-bottom: 4px;'>{b}</li>")
            html_parts.append("</ul>")

        if decisions:
            html_parts.append("<b style='color: #7ee787;'>Agreed Decisions:</b><ul style='margin-top: 4px;'>")
            for d in decisions:
                html_parts.append(f"<li style='margin-bottom: 4px;'>{d}</li>")
            html_parts.append("</ul>")

        if not html_parts:
            self.summary_browser.setHtml(
                "<p style='color: #8b949e; font-style: italic;'>Meeting summary will generate dynamically as participants speak...</p>"
            )
        else:
            self.summary_browser.setHtml("".join(html_parts))
            self.summary_updated_lbl.setText(f"Updated {time.strftime('%H:%M:%S')}")

        # Update action items checklist
        while self.actions_layout.count() > 1:
            item = self.actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        action_items = summary_data.get("action_items", [])
        for act in action_items:
            row = ActionItemRow(act, parent=self.actions_container)
            self.actions_layout.insertWidget(self.actions_layout.count() - 1, row)

        self.actions_count_lbl.setText(f"{len(action_items)} action items")

    @Slot(str)
    def _on_status_changed(self, status: str):
        self.status_bar.showMessage(f"Status: {status} | Provider: {self.pipeline.provider_info.active_provider}")

    @Slot(str)
    def _on_error(self, err: str):
        QMessageBox.warning(self, "Hardware Abstraction Layer", err)

    def _toggle_meeting(self):
        if not self.pipeline._is_running:
            self.pipeline.start()
            self.btn_start.setText("⏹ Stop Meeting")
            self.btn_start.setProperty("class", "dangerButton")
            self.btn_start.setStyle(self.btn_start.style())
            self.timer.start(1000)
        else:
            self.pipeline.stop()
            self.btn_start.setText("▶ Start Meeting")
            self.btn_start.setProperty("class", "accentButton")
            self.btn_start.setStyle(self.btn_start.style())
            self.timer.stop()
            self.vu_meter.setValue(0)

    def _update_timer(self):
        self.meeting_seconds += 1
        formatted = str(timedelta(seconds=self.meeting_seconds))
        if len(formatted) == 7:
            formatted = "0" + formatted
        self.timer_display.setText(formatted)

    def _clear_session(self):
        confirm = QMessageBox.question(
            self,
            "Clear Session",
            "Are you sure you want to clear current transcript and action items?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.pipeline.clear()
            self.meeting_seconds = 0
            self.timer_display.setText("00:00:00")
            self.vu_meter.setValue(0)
            self.utterance_count_lbl.setText("0 utterances")

            while self.transcript_layout.count() > 1:
                item = self.transcript_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            while self.actions_layout.count() > 1:
                item = self.actions_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.actions_count_lbl.setText("0 items")
            self.summary_browser.setHtml(
                "<p style='color: #8b949e; font-style: italic;'>Meeting summary will generate dynamically as participants speak...</p>"
            )

    def _export_dialog(self):
        if not self.pipeline.transcript_history:
            QMessageBox.information(self, "Export", "No transcript data available to export yet.")
            return

        default_name = f"meeting_notes_{time.strftime('%Y%m%d_%H%M%S')}.md"
        default_path = str(Path(self.config.app.export_dir) / default_name)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Meeting Notes",
            default_path,
            "Markdown Files (*.md);;JSON Files (*.json)"
        )

        if filepath:
            fmt = "json" if filepath.endswith(".json") else "md"
            saved_path = self.pipeline.export_notes(filepath, fmt=fmt)
            QMessageBox.information(
                self, "Export Successful", f"Meeting minutes saved offline to:\n{saved_path}"
            )

    def _show_config_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Hardware Abstraction Layer & Provider Diagnostics")
        dialog.resize(600, 460)
        layout = QVBoxLayout(dialog)

        info = QTextBrowser()
        info.setStyleSheet("background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 12px;")
        
        p = self.pipeline.provider_info
        fallback_html = (
            f"<li><b>Fallback Active:</b> <span style='color: #e3b341;'>YES ({p.fallback_reason})</span></li>"
            if p.is_fallback else "<li><b>Fallback Active:</b> <span style='color: #3fb950;'>NO (Direct execution)</span></li>"
        )

        html = f"""
        <h3 style='color: #58a6ff; margin-top: 0;'>Hardware Abstraction Layer</h3>
        <ul>
            <li><b>Host Platform:</b> {p.device_platform}</li>
            <li><b>Requested Backend:</b> <code>{p.requested_backend}</code></li>
            <li><b>Active Provider:</b> <code style='color: #3fb950;'>{p.active_provider}</code></li>
            <li><b>Display Status:</b> <b>{p.display_status}</b></li>
            {fallback_html}
            <li><b>Qualcomm HTP Backend:</b> {p.qnn_backend_path}</li>
            <li><b>Available ONNX Providers:</b> {', '.join(p.available_providers)}</li>
        </ul>
        <h3 style='color: #58a6ff;'>Model Artifact Paths (Target: {p.active_backend.upper()})</h3>
        <ul>
            <li><b>Whisper STT:</b> <code>{self.config.get_stt_model_path(p.active_backend)}</code></li>
            <li><b>Speaker Diarization:</b> <code>{self.config.get_diarization_model_path(p.active_backend)}</code></li>
            <li><b>LLM Summarizer:</b> <code>{self.config.get_summarizer_model_path(p.active_backend)}</code></li>
        </ul>
        <h3 style='color: #58a6ff;'>Zero Cloud Offline Guarantee</h3>
        <p style='color: #8b949e;'>All microphone capture, speech-to-text, speaker clustering, and summarization run 100% locally through ONNX Runtime. No audio or text leaves this device.</p>
        """
        info.setHtml(html)
        layout.addWidget(info)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.exec()

    def closeEvent(self, event):
        """Clean shutdown when closing window."""
        self.pipeline.stop()
        event.accept()