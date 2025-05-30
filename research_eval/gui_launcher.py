import sys
from PyQt6.QtWidgets import QApplication
from research_eval.gui.main_window import MainWindow

def run_gui():
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    run_gui()
