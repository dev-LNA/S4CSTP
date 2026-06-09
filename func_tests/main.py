from PyQt6 import QtWidgets

from .gui import GUI

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = GUI()
    window.show()
    app.exec()
