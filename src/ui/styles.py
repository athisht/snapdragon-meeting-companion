"""
UI Stylesheet and Theme Definitions
Modern dark theme with Windows 11 Fluent aesthetics and Snapdragon cyan accents.
"""

DARK_THEME_QSS = """
/* Global Window Styling */
QMainWindow, QDialog {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QWidget {
    color: #e6edf3;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #161b22;
    width: 8px;
    border-radius: 4px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Card Containers */
QFrame.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px;
}

QFrame.headerCard {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 10px 16px;
}

/* Buttons */
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #363b42;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background-color: #161b22;
}

QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

QPushButton.primaryButton {
    background-color: #0969da;
    border: 1px solid #1f6feb;
    color: #ffffff;
}

QPushButton.primaryButton:hover {
    background-color: #1f6feb;
}

QPushButton.accentButton {
    background-color: #0078d4;
    border: 1px solid #00d4ff;
    color: #ffffff;
}

QPushButton.accentButton:hover {
    background-color: #0086ea;
}

QPushButton.dangerButton {
    background-color: #da3633;
    border: 1px solid #f85149;
    color: #ffffff;
}

QPushButton.dangerButton:hover {
    background-color: #b62324;
}

/* Labels and Headers */
QLabel.titleLabel {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
}

QLabel.sectionHeader {
    font-size: 14px;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 0.5px;
}

QLabel.badgeNPU {
    background-color: #0d3826;
    color: #3fb950;
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
}

QLabel.badgeCPU {
    background-color: #0c2d6b;
    color: #58a6ff;
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
}

QLabel.badgeFallback {
    background-color: #382405;
    color: #e3b341;
    border: 1px solid #9e6a03;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
}

QLabel.badgeMock {
    background-color: #2b1d3a;
    color: #d2a8ff;
    border: 1px solid #8957e5;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.timerLabel {
    font-family: 'Consolas', monospace;
    font-size: 16px;
    font-weight: bold;
    color: #3fb950;
}

/* Progress Bar (VU Meter) */
QProgressBar {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
    height: 12px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #238636, stop:0.7 #2ea043, stop:1.0 #00d4ff);
    border-radius: 3px;
}

/* Text Editors / Lists */
QTextEdit, QListWidget {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
    padding: 8px;
    font-size: 13px;
}

/* Splitter */
QSplitter::handle {
    background-color: #21262d;
    width: 3px;
}

QSplitter::handle:hover {
    background-color: #00d4ff;
}

/* Status Bar */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #30363d;
    font-size: 11px;
}
"""

SPEAKER_COLORS = [
    {"bg": "#0e3a5a", "border": "#1f6feb", "text": "#79c0ff"},  # Speaker 1 (Cyan/Blue)
    {"bg": "#123d24", "border": "#238636", "text": "#7ee787"},  # Speaker 2 (Green)
    {"bg": "#3e240a", "border": "#9e6a03", "text": "#f0b72f"},  # Speaker 3 (Amber)
    {"bg": "#3c1642", "border": "#8957e5", "text": "#d2a8ff"},  # Speaker 4 (Purple)
    {"bg": "#431418", "border": "#da3633", "text": "#ffa198"},  # Speaker 5 (Coral)
]