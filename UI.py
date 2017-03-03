from PyQt5 import QtCore, QtGui, QtWidgets

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(503, 493)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(40, 60, 321, 146))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.txtLearningRate = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtLearningRate.setObjectName("txtLearningRate")
        self.gridLayout.addWidget(self.txtLearningRate, 4, 1, 1, 1)
        self.lblLearningRate = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblLearningRate.setObjectName("lblLearningRate")
        self.gridLayout.addWidget(self.lblLearningRate, 4, 0, 1, 1)
        self.lblOptimizer = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblOptimizer.setObjectName("lblOptimizer")
        self.gridLayout.addWidget(self.lblOptimizer, 6, 0, 1, 1)
        self.txtBatchSize = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtBatchSize.setObjectName("txtBatchSize")
        self.gridLayout.addWidget(self.txtBatchSize, 1, 1, 1, 1)
        self.lblBatchSize = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblBatchSize.setObjectName("lblBatchSize")
        self.gridLayout.addWidget(self.lblBatchSize, 1, 0, 1, 1)
        self.cbxOptimizer = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.cbxOptimizer.setObjectName("cbxOptimizer")
        self.cbxOptimizer.addItem("")
        self.cbxOptimizer.addItem("")
        self.cbxOptimizer.addItem("")
        self.gridLayout.addWidget(self.cbxOptimizer, 6, 1, 1, 1)
        self.lblNumEpochs = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblNumEpochs.setObjectName("lblNumEpochs")
        self.gridLayout.addWidget(self.lblNumEpochs, 5, 0, 1, 1)
        self.txtNumEpochs = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtNumEpochs.setObjectName("txtNumEpochs")
        self.gridLayout.addWidget(self.txtNumEpochs, 5, 1, 1, 1)
        self.radCIFAR10 = QtWidgets.QRadioButton(self.centralwidget)
        self.radCIFAR10.setGeometry(QtCore.QRect(40, 20, 82, 17))
        self.radCIFAR10.setObjectName("radCIFAR10")
        self.prgTrainingProgress = QtWidgets.QProgressBar(self.centralwidget)
        self.prgTrainingProgress.setGeometry(QtCore.QRect(50, 290, 319, 21))
        self.prgTrainingProgress.setProperty("value", 24)
        self.prgTrainingProgress.setObjectName("prgTrainingProgress")
        self.btnTrainNetwork = QtWidgets.QPushButton(self.centralwidget)
        self.btnTrainNetwork.setGeometry(QtCore.QRect(140, 220, 111, 31))
        self.btnTrainNetwork.setObjectName("btnTrainNetwork")
        self.lblTrainingProgress = QtWidgets.QLabel(self.centralwidget)
        self.lblTrainingProgress.setGeometry(QtCore.QRect(150, 270, 101, 16))
        self.lblTrainingProgress.setObjectName("lblTrainingProgress")
        self.lblTestAccuracy = QtWidgets.QLabel(self.centralwidget)
        self.lblTestAccuracy.setGeometry(QtCore.QRect(40, 340, 111, 16))
        self.lblTestAccuracy.setObjectName("lblTestAccuracy")
        self.txtTestAccuracy = QtWidgets.QLineEdit(self.centralwidget)
        self.txtTestAccuracy.setGeometry(QtCore.QRect(140, 340, 91, 20))
        self.txtTestAccuracy.setObjectName("txtTestAccuracy")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 503, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "First CNN GUI"))
        self.lblLearningRate.setText(_translate("MainWindow", "Learning Rate:"))
        self.lblOptimizer.setText(_translate("MainWindow", "Optimizer:"))
        self.lblBatchSize.setText(_translate("MainWindow", "Batch Size:"))
        self.cbxOptimizer.setItemText(0, _translate("MainWindow", "Gradient Descent Optimizer"))
        self.cbxOptimizer.setItemText(1, _translate("MainWindow", "Adam Optimizer"))
        self.cbxOptimizer.setItemText(2, _translate("MainWindow", "AdaGrad Optimizer"))
        self.lblNumEpochs.setText(_translate("MainWindow", "Number of Epochs:"))
        self.radCIFAR10.setText(_translate("MainWindow", "CIFAR-10"))
        self.btnTrainNetwork.setText(_translate("MainWindow", "Train Network"))
        self.lblTrainingProgress.setText(_translate("MainWindow", "Training Progress"))
        self.lblTestAccuracy.setText(_translate("MainWindow", "Test Set Accuracy:"))