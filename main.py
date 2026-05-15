import sys
from PySide6.QtWidgets import (
    QApplication, QMessageBox, QLineEdit,
    QDialog, QVBoxLayout, QLabel, QPushButton
)

from gui.main_window import MainWindow


class SecurityDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security Key Required")
        self.setModal(True)
        self.setFixedSize(300, 120)

        self.input = QLineEdit()
        self.input.setEchoMode(QLineEdit.Password)
        self.input.setPlaceholderText("Enter security key")

        self.ok_button = QPushButton("Unlock")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Please enter the security key to continue:"))
        layout.addWidget(self.input)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_key(self):
        if self.exec() == QDialog.Accepted:  # exec_() → exec()
            return self.input.text()
        return None


def verify_security_key():
    correct_key = "indianarmy"
    for attempt in range(3):
        dialog = SecurityDialog()
        key = dialog.get_key()
        if key is None:
            return False
        if key == correct_key:
            return True
        QMessageBox.warning(None, "Incorrect Key", "Wrong key! Please try again.")
    return False


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Object Detection and Classification")
    app.setApplicationVersion("1.0")

    if not verify_security_key():
        sys.exit(0)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())  # exec_() → exec()


if __name__ == "__main__":
    main()
