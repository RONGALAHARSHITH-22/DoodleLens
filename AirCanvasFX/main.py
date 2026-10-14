"""
DoodleLens - Entry Point
"""

import sys
import os

# Ensure the project directory is on the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def check_dependencies():
    """Check all required packages are installed."""
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    try:
        import mediapipe
    except ImportError:
        missing.append("mediapipe")
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import PIL
    except ImportError:
        missing.append("pillow")

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\nInstall all at once:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


def main():
    check_dependencies()

    from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap, QFont, QColor

    app = QApplication(sys.argv)
    app.setApplicationName("DoodleLens")
    app.setOrganizationName("DoodleLens")

    # High DPI is enabled by default in PyQt6 6.4+

    # Show splash
    splash_widget = QLabel(
        "✨ DoodleLens\n\nLoading modules...",
        alignment=Qt.AlignmentFlag.AlignCenter
    )
    splash_widget.setFixedSize(500, 300)
    splash_widget.setWindowFlags(
        Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
    )
    splash_widget.setStyleSheet("""
        QLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0d0d1a, stop:0.5 #140d2e, stop:1 #0d1a0d);
            color: #c084fc;
            font-size: 22px;
            font-weight: bold;
            font-family: 'Segoe UI', Arial, sans-serif;
            border: 2px solid #9b5cf6;
            border-radius: 12px;
        }
    """)
    splash_widget.show()
    app.processEvents()

    # Import and launch main window
    from src.ui.main_window import MainWindow

    window = MainWindow(base_dir=BASE_DIR)

    def show_main():
        splash_widget.close()
        window.show()

    QTimer.singleShot(700, show_main)


    sys.exit(app.exec())


if __name__ == "__main__":
    main()
