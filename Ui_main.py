from PyQt5 import QtCore, QtWidgets


class Ui_mainwindow(object):
    def __init__(self):
        self.mainwindow = QtWidgets.QMainWindow()
        self.central_widget = QtWidgets.QWidget(self.mainwindow)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.image_label = QtWidgets.QLabel(self.central_widget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton(self.central_widget)
        self.line_path = QtWidgets.QLineEdit(self.central_widget)
        self.btn_browser = QtWidgets.QPushButton(self.central_widget)
        self.status_bar = QtWidgets.QStatusBar(self.mainwindow)

    def setupUi(self):
        self.mainwindow = QtWidgets.QMainWindow()
        self.central_widget = QtWidgets.QWidget(self.mainwindow)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.central_widget)
        self.image_label = QtWidgets.QLabel(self.central_widget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton(self.central_widget)
        self.line_path = QtWidgets.QLineEdit(self.central_widget)
        self.btn_browser = QtWidgets.QPushButton(self.central_widget)
        self.status_bar = QtWidgets.QStatusBar(self.mainwindow)

        self.mainwindow.setObjectName("mainwindow")
        self.central_widget.setObjectName("central_widget")
        self.verticalLayout.setObjectName("verticalLayout")

        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)

        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_start.setObjectName("btn_start")
        self.line_path.setObjectName("line_path")
        self.btn_browser.setObjectName("btn_browser")

        self.horizontalLayout.addWidget(self.btn_start)
        self.horizontalLayout.addWidget(self.line_path)
        self.horizontalLayout.addWidget(self.btn_browser)

        self.verticalLayout.addLayout(self.horizontalLayout)
        self.mainwindow.setCentralWidget(self.central_widget)

        self.status_bar.setObjectName("status_bar")
        self.mainwindow.setStatusBar(self.status_bar)

        self.retranslateUi(self.mainwindow)
        QtCore.QMetaObject.connectSlotsByName(self.mainwindow)

    def retranslateUi(self, mainwindow):
        _translate = QtCore.QCoreApplication.translate
        mainwindow.setWindowTitle(_translate("mainwindow", "Omniverse"))
        self.btn_start.setText(_translate("mainwindow", "Start"))
        self.btn_browser.setText(_translate("mainwindow", "Browser"))

    def showUi(self):
        self.mainwindow.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_mainwindow()
    ui.setupUi()
    ui.showUi()
    sys.exit(app.exec_())
