import sys

from PyQt6 import QtWidgets

from func_tests.gui import GUI

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    app.exec()
