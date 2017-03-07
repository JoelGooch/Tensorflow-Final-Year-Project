# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CNN_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.8
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1198, 802)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 400, 1171, 361))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.txtOutputLog = QtWidgets.QTextEdit(self.groupBox_2)
        self.txtOutputLog.setGeometry(QtCore.QRect(420, 40, 721, 311))
        self.txtOutputLog.setObjectName("txtOutputLog")
        self.btnCancelTraining = QtWidgets.QPushButton(self.groupBox_2)
        self.btnCancelTraining.setGeometry(QtCore.QRect(220, 190, 131, 41))
        self.btnCancelTraining.setObjectName("btnCancelTraining")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(40, 30, 321, 146))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.txtNumEpochs = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtNumEpochs.setObjectName("txtNumEpochs")
        self.gridLayout.addWidget(self.txtNumEpochs, 5, 1, 1, 1)
        self.lblNumEpochs = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblNumEpochs.setObjectName("lblNumEpochs")
        self.gridLayout.addWidget(self.lblNumEpochs, 5, 0, 1, 1)
        self.lblBatchSize = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblBatchSize.setObjectName("lblBatchSize")
        self.gridLayout.addWidget(self.lblBatchSize, 1, 0, 1, 1)
        self.lblLearningRate = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblLearningRate.setObjectName("lblLearningRate")
        self.gridLayout.addWidget(self.lblLearningRate, 4, 0, 1, 1)
        self.txtLearningRate = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtLearningRate.setObjectName("txtLearningRate")
        self.gridLayout.addWidget(self.txtLearningRate, 4, 1, 1, 1)
        self.lblOptimizer = QtWidgets.QLabel(self.gridLayoutWidget)
        self.lblOptimizer.setObjectName("lblOptimizer")
        self.gridLayout.addWidget(self.lblOptimizer, 6, 0, 1, 1)
        self.cbxOptimizer = QtWidgets.QComboBox(self.gridLayoutWidget)
        self.cbxOptimizer.setObjectName("cbxOptimizer")
        self.cbxOptimizer.addItem("")
        self.cbxOptimizer.addItem("")
        self.cbxOptimizer.addItem("")
        self.gridLayout.addWidget(self.cbxOptimizer, 6, 1, 1, 1)
        self.txtBatchSize = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.txtBatchSize.setObjectName("txtBatchSize")
        self.gridLayout.addWidget(self.txtBatchSize, 1, 1, 1, 1)
        self.txtTestAccuracy = QtWidgets.QLineEdit(self.groupBox_2)
        self.txtTestAccuracy.setEnabled(False)
        self.txtTestAccuracy.setGeometry(QtCore.QRect(200, 330, 91, 20))
        self.txtTestAccuracy.setReadOnly(True)
        self.txtTestAccuracy.setObjectName("txtTestAccuracy")
        self.prgTrainingProgress = QtWidgets.QProgressBar(self.groupBox_2)
        self.prgTrainingProgress.setGeometry(QtCore.QRect(38, 250, 331, 21))
        self.prgTrainingProgress.setProperty("value", 0)
        self.prgTrainingProgress.setObjectName("prgTrainingProgress")
        self.lblTrainingProgress = QtWidgets.QLabel(self.groupBox_2)
        self.lblTrainingProgress.setGeometry(QtCore.QRect(150, 280, 101, 16))
        self.lblTrainingProgress.setObjectName("lblTrainingProgress")
        self.lblTestAccuracy = QtWidgets.QLabel(self.groupBox_2)
        self.lblTestAccuracy.setGeometry(QtCore.QRect(100, 330, 111, 16))
        self.lblTestAccuracy.setObjectName("lblTestAccuracy")
        self.btnTrainNetwork = QtWidgets.QPushButton(self.groupBox_2)
        self.btnTrainNetwork.setGeometry(QtCore.QRect(50, 190, 131, 41))
        self.btnTrainNetwork.setObjectName("btnTrainNetwork")
        self.lblOutputLog = QtWidgets.QLabel(self.groupBox_2)
        self.lblOutputLog.setGeometry(QtCore.QRect(420, 20, 91, 16))
        self.lblOutputLog.setObjectName("lblOutputLog")
        self.grpNetworkModel = QtWidgets.QGroupBox(self.centralwidget)
        self.grpNetworkModel.setGeometry(QtCore.QRect(10, 20, 1171, 371))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grpNetworkModel.sizePolicy().hasHeightForWidth())
        self.grpNetworkModel.setSizePolicy(sizePolicy)
        self.grpNetworkModel.setObjectName("grpNetworkModel")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.grpNetworkModel)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(160, 20, 961, 341))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cbxLayerSelect = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.cbxLayerSelect.setObjectName("cbxLayerSelect")
        self.cbxLayerSelect.addItem("")
        self.cbxLayerSelect.addItem("")
        self.cbxLayerSelect.addItem("")
        self.cbxLayerSelect.addItem("")
        self.cbxLayerSelect.addItem("")
        self.cbxLayerSelect.addItem("")
        self.horizontalLayout.addWidget(self.cbxLayerSelect)
        self.btnAddLayer = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.btnAddLayer.setObjectName("btnAddLayer")
        self.horizontalLayout.addWidget(self.btnAddLayer)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.tabMenuLayer = QtWidgets.QTabWidget(self.horizontalLayoutWidget_2)
        self.tabMenuLayer.setEnabled(False)
        self.tabMenuLayer.setObjectName("tabMenuLayer")
        self.tabConvLayer = QtWidgets.QWidget()
        self.tabConvLayer.setObjectName("tabConvLayer")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tabConvLayer)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(30, 20, 391, 186))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.cbxConvPadding = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.cbxConvPadding.setObjectName("cbxConvPadding")
        self.cbxConvPadding.addItem("")
        self.cbxConvPadding.addItem("")
        self.gridLayout_2.addWidget(self.cbxConvPadding, 7, 1, 1, 1)
        self.lblConvStDev = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvStDev.setObjectName("lblConvStDev")
        self.gridLayout_2.addWidget(self.lblConvStDev, 6, 0, 1, 1)
        self.cbxConvoWeightInit = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.cbxConvoWeightInit.setObjectName("cbxConvoWeightInit")
        self.cbxConvoWeightInit.addItem("")
        self.gridLayout_2.addWidget(self.cbxConvoWeightInit, 5, 1, 1, 1)
        self.lblConvWeightInit = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvWeightInit.setObjectName("lblConvWeightInit")
        self.gridLayout_2.addWidget(self.lblConvWeightInit, 5, 0, 1, 1)
        self.lblConvKernelSize = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvKernelSize.setObjectName("lblConvKernelSize")
        self.gridLayout_2.addWidget(self.lblConvKernelSize, 0, 0, 1, 1)
        self.lblConvOutputFilters = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvOutputFilters.setObjectName("lblConvOutputFilters")
        self.gridLayout_2.addWidget(self.lblConvOutputFilters, 3, 0, 1, 1)
        self.lblConvStride = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvStride.setObjectName("lblConvStride")
        self.gridLayout_2.addWidget(self.lblConvStride, 1, 0, 1, 1)
        self.lblConvPadding = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvPadding.setObjectName("lblConvPadding")
        self.gridLayout_2.addWidget(self.lblConvPadding, 7, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.txtConvStDev = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvStDev.setObjectName("txtConvStDev")
        self.horizontalLayout_4.addWidget(self.txtConvStDev)
        spacerItem1 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 6, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.txtConvStride1 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvStride1.setObjectName("txtConvStride1")
        self.horizontalLayout_3.addWidget(self.txtConvStride1)
        self.txtConvStride2 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvStride2.setObjectName("txtConvStride2")
        self.horizontalLayout_3.addWidget(self.txtConvStride2)
        self.txtConvStride3 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvStride3.setObjectName("txtConvStride3")
        self.horizontalLayout_3.addWidget(self.txtConvStride3)
        self.txtConvStride4 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvStride4.setObjectName("txtConvStride4")
        self.horizontalLayout_3.addWidget(self.txtConvStride4)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 1, 1, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.txtConvOutputFilters = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvOutputFilters.setObjectName("txtConvOutputFilters")
        self.horizontalLayout_5.addWidget(self.txtConvOutputFilters)
        spacerItem2 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 3, 1, 1, 1)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.txtConvKernelSize = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.txtConvKernelSize.setObjectName("txtConvKernelSize")
        self.horizontalLayout_12.addWidget(self.txtConvKernelSize)
        spacerItem3 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem3)
        self.gridLayout_2.addLayout(self.horizontalLayout_12, 0, 1, 1, 1)
        self.lblConvActFunction = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.lblConvActFunction.setObjectName("lblConvActFunction")
        self.gridLayout_2.addWidget(self.lblConvActFunction, 2, 0, 1, 1)
        self.cbxConvActFunction = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.cbxConvActFunction.setObjectName("cbxConvActFunction")
        self.cbxConvActFunction.addItem("")
        self.cbxConvActFunction.addItem("")
        self.gridLayout_2.addWidget(self.cbxConvActFunction, 2, 1, 1, 1)
        self.btnAddConvLayer = QtWidgets.QPushButton(self.tabConvLayer)
        self.btnAddConvLayer.setGeometry(QtCore.QRect(160, 220, 131, 31))
        self.btnAddConvLayer.setObjectName("btnAddConvLayer")
        self.tabMenuLayer.addTab(self.tabConvLayer, "")
        self.tabMaxPooling = QtWidgets.QWidget()
        self.tabMaxPooling.setObjectName("tabMaxPooling")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.tabMaxPooling)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(30, 20, 391, 91))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lblPoolStride = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.lblPoolStride.setObjectName("lblPoolStride")
        self.gridLayout_3.addWidget(self.lblPoolStride, 1, 0, 1, 1)
        self.lblPoolPadding = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.lblPoolPadding.setObjectName("lblPoolPadding")
        self.gridLayout_3.addWidget(self.lblPoolPadding, 2, 0, 1, 1)
        self.lblPoolKernelSize = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.lblPoolKernelSize.setObjectName("lblPoolKernelSize")
        self.gridLayout_3.addWidget(self.lblPoolKernelSize, 0, 0, 1, 1)
        self.cbxPoolPadding = QtWidgets.QComboBox(self.gridLayoutWidget_3)
        self.cbxPoolPadding.setObjectName("cbxPoolPadding")
        self.gridLayout_3.addWidget(self.cbxPoolPadding, 2, 1, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.txtPoolStride1 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.txtPoolStride1.setObjectName("txtPoolStride1")
        self.horizontalLayout_6.addWidget(self.txtPoolStride1)
        self.txtPoolStride2 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.txtPoolStride2.setObjectName("txtPoolStride2")
        self.horizontalLayout_6.addWidget(self.txtPoolStride2)
        self.txtPoolStride3 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.txtPoolStride3.setObjectName("txtPoolStride3")
        self.horizontalLayout_6.addWidget(self.txtPoolStride3)
        self.txtPoolStride4 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.txtPoolStride4.setObjectName("txtPoolStride4")
        self.horizontalLayout_6.addWidget(self.txtPoolStride4)
        self.gridLayout_3.addLayout(self.horizontalLayout_6, 1, 1, 1, 1)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.txtPoolKernelSize = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.txtPoolKernelSize.setObjectName("txtPoolKernelSize")
        self.horizontalLayout_13.addWidget(self.txtPoolKernelSize)
        spacerItem4 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem4)
        self.gridLayout_3.addLayout(self.horizontalLayout_13, 0, 1, 1, 1)
        self.btnAddMaxPool = QtWidgets.QPushButton(self.tabMaxPooling)
        self.btnAddMaxPool.setGeometry(QtCore.QRect(160, 130, 131, 31))
        self.btnAddMaxPool.setObjectName("btnAddMaxPool")
        self.tabMenuLayer.addTab(self.tabMaxPooling, "")
        self.tabDropout = QtWidgets.QWidget()
        self.tabDropout.setObjectName("tabDropout")
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.tabDropout)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(30, 20, 391, 51))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.lblPoolKeepRate = QtWidgets.QLabel(self.gridLayoutWidget_5)
        self.lblPoolKeepRate.setObjectName("lblPoolKeepRate")
        self.gridLayout_5.addWidget(self.lblPoolKeepRate, 0, 0, 1, 1)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.txtPoolKeepRate = QtWidgets.QLineEdit(self.gridLayoutWidget_5)
        self.txtPoolKeepRate.setObjectName("txtPoolKeepRate")
        self.horizontalLayout_16.addWidget(self.txtPoolKeepRate)
        spacerItem5 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem5)
        self.gridLayout_5.addLayout(self.horizontalLayout_16, 0, 1, 1, 1)
        self.btnAddDropout = QtWidgets.QPushButton(self.tabDropout)
        self.btnAddDropout.setGeometry(QtCore.QRect(170, 90, 131, 31))
        self.btnAddDropout.setObjectName("btnAddDropout")
        self.tabMenuLayer.addTab(self.tabDropout, "")
        self.tabNormalizing = QtWidgets.QWidget()
        self.tabNormalizing.setObjectName("tabNormalizing")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.tabNormalizing)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(30, 20, 391, 121))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lblNormBias = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.lblNormBias.setObjectName("lblNormBias")
        self.gridLayout_4.addWidget(self.lblNormBias, 0, 0, 1, 1)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.txtNormAlpha = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.txtNormAlpha.setObjectName("txtNormAlpha")
        self.horizontalLayout_9.addWidget(self.txtNormAlpha)
        spacerItem6 = QtWidgets.QSpacerItem(280, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem6)
        self.gridLayout_4.addLayout(self.horizontalLayout_9, 1, 1, 1, 1)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.txtNormBeta = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.txtNormBeta.setObjectName("txtNormBeta")
        self.horizontalLayout_11.addWidget(self.txtNormBeta)
        spacerItem7 = QtWidgets.QSpacerItem(280, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem7)
        self.gridLayout_4.addLayout(self.horizontalLayout_11, 2, 1, 1, 1)
        self.lblNormAlpha = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.lblNormAlpha.setObjectName("lblNormAlpha")
        self.gridLayout_4.addWidget(self.lblNormAlpha, 1, 0, 1, 1)
        self.lblNormBeta = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.lblNormBeta.setObjectName("lblNormBeta")
        self.gridLayout_4.addWidget(self.lblNormBeta, 2, 0, 1, 1)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.txtNormBias = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.txtNormBias.setObjectName("txtNormBias")
        self.horizontalLayout_10.addWidget(self.txtNormBias)
        spacerItem8 = QtWidgets.QSpacerItem(280, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem8)
        self.gridLayout_4.addLayout(self.horizontalLayout_10, 0, 1, 1, 1)
        self.btnAddNormalizing = QtWidgets.QPushButton(self.tabNormalizing)
        self.btnAddNormalizing.setGeometry(QtCore.QRect(160, 160, 131, 31))
        self.btnAddNormalizing.setObjectName("btnAddNormalizing")
        self.tabMenuLayer.addTab(self.tabNormalizing, "")
        self.tabFullyConnected = QtWidgets.QWidget()
        self.tabFullyConnected.setObjectName("tabFullyConnected")
        self.gridLayoutWidget_6 = QtWidgets.QWidget(self.tabFullyConnected)
        self.gridLayoutWidget_6.setGeometry(QtCore.QRect(30, 20, 391, 131))
        self.gridLayoutWidget_6.setObjectName("gridLayoutWidget_6")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.gridLayoutWidget_6)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lblFCWeightInit = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.lblFCWeightInit.setObjectName("lblFCWeightInit")
        self.gridLayout_6.addWidget(self.lblFCWeightInit, 3, 0, 1, 1)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.txtFCStDev = QtWidgets.QLineEdit(self.gridLayoutWidget_6)
        self.txtFCStDev.setObjectName("txtFCStDev")
        self.horizontalLayout_14.addWidget(self.txtFCStDev)
        spacerItem9 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem9)
        self.gridLayout_6.addLayout(self.horizontalLayout_14, 4, 1, 1, 1)
        self.lblFCNumOutputNodes = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.lblFCNumOutputNodes.setObjectName("lblFCNumOutputNodes")
        self.gridLayout_6.addWidget(self.lblFCNumOutputNodes, 1, 0, 1, 1)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.txtFCNumOutputNodes = QtWidgets.QLineEdit(self.gridLayoutWidget_6)
        self.txtFCNumOutputNodes.setObjectName("txtFCNumOutputNodes")
        self.horizontalLayout_17.addWidget(self.txtFCNumOutputNodes)
        spacerItem10 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_17.addItem(spacerItem10)
        self.gridLayout_6.addLayout(self.horizontalLayout_17, 1, 1, 1, 1)
        self.lblFCStDev = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.lblFCStDev.setObjectName("lblFCStDev")
        self.gridLayout_6.addWidget(self.lblFCStDev, 4, 0, 1, 1)
        self.cbxFCWeightInit = QtWidgets.QComboBox(self.gridLayoutWidget_6)
        self.cbxFCWeightInit.setObjectName("cbxFCWeightInit")
        self.cbxFCWeightInit.addItem("")
        self.gridLayout_6.addWidget(self.cbxFCWeightInit, 3, 1, 1, 1)
        self.cbxFCActFunction = QtWidgets.QComboBox(self.gridLayoutWidget_6)
        self.cbxFCActFunction.setObjectName("cbxFCActFunction")
        self.cbxFCActFunction.addItem("")
        self.cbxFCActFunction.addItem("")
        self.gridLayout_6.addWidget(self.cbxFCActFunction, 0, 1, 1, 1)
        self.lblFCActFunction = QtWidgets.QLabel(self.gridLayoutWidget_6)
        self.lblFCActFunction.setObjectName("lblFCActFunction")
        self.gridLayout_6.addWidget(self.lblFCActFunction, 0, 0, 1, 1)
        self.btnAddFullyConn = QtWidgets.QPushButton(self.tabFullyConnected)
        self.btnAddFullyConn.setGeometry(QtCore.QRect(160, 170, 131, 31))
        self.btnAddFullyConn.setObjectName("btnAddFullyConn")
        self.tabMenuLayer.addTab(self.tabFullyConnected, "")
        self.tabOutput = QtWidgets.QWidget()
        self.tabOutput.setObjectName("tabOutput")
        self.gridLayoutWidget_7 = QtWidgets.QWidget(self.tabOutput)
        self.gridLayoutWidget_7.setGeometry(QtCore.QRect(30, 20, 391, 101))
        self.gridLayoutWidget_7.setObjectName("gridLayoutWidget_7")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.gridLayoutWidget_7)
        self.gridLayout_7.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.lblOutputStDev = QtWidgets.QLabel(self.gridLayoutWidget_7)
        self.lblOutputStDev.setObjectName("lblOutputStDev")
        self.gridLayout_7.addWidget(self.lblOutputStDev, 3, 0, 1, 1)
        self.cbxOutputWeightInit = QtWidgets.QComboBox(self.gridLayoutWidget_7)
        self.cbxOutputWeightInit.setObjectName("cbxOutputWeightInit")
        self.cbxOutputWeightInit.addItem("")
        self.gridLayout_7.addWidget(self.cbxOutputWeightInit, 2, 1, 1, 1)
        self.lblOutputActFunction = QtWidgets.QLabel(self.gridLayoutWidget_7)
        self.lblOutputActFunction.setObjectName("lblOutputActFunction")
        self.gridLayout_7.addWidget(self.lblOutputActFunction, 0, 0, 1, 1)
        self.cbxOutputActFunction = QtWidgets.QComboBox(self.gridLayoutWidget_7)
        self.cbxOutputActFunction.setObjectName("cbxOutputActFunction")
        self.cbxOutputActFunction.addItem("")
        self.cbxOutputActFunction.addItem("")
        self.gridLayout_7.addWidget(self.cbxOutputActFunction, 0, 1, 1, 1)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.txtOutputStDev = QtWidgets.QLineEdit(self.gridLayoutWidget_7)
        self.txtOutputStDev.setObjectName("txtOutputStDev")
        self.horizontalLayout_15.addWidget(self.txtOutputStDev)
        spacerItem11 = QtWidgets.QSpacerItem(210, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem11)
        self.gridLayout_7.addLayout(self.horizontalLayout_15, 3, 1, 1, 1)
        self.lblOutputWeightInit = QtWidgets.QLabel(self.gridLayoutWidget_7)
        self.lblOutputWeightInit.setObjectName("lblOutputWeightInit")
        self.gridLayout_7.addWidget(self.lblOutputWeightInit, 2, 0, 1, 1)
        self.btnAddOutput = QtWidgets.QPushButton(self.tabOutput)
        self.btnAddOutput.setGeometry(QtCore.QRect(160, 150, 131, 31))
        self.btnAddOutput.setObjectName("btnAddOutput")
        self.tabMenuLayer.addTab(self.tabOutput, "")
        self.verticalLayout.addWidget(self.tabMenuLayer)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        spacerItem12 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem12)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.tblNetworkModel = QtWidgets.QTableView(self.horizontalLayoutWidget_2)
        self.tblNetworkModel.setObjectName("tblNetworkModel")
        self.horizontalLayout_8.addWidget(self.tblNetworkModel)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.btnValidateNetwork = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.btnValidateNetwork.setObjectName("btnValidateNetwork")
        self.verticalLayout_3.addWidget(self.btnValidateNetwork)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.grpDataSet = QtWidgets.QGroupBox(self.grpNetworkModel)
        self.grpDataSet.setGeometry(QtCore.QRect(20, 40, 120, 91))
        self.grpDataSet.setObjectName("grpDataSet")
        self.radCIFAR10 = QtWidgets.QRadioButton(self.grpDataSet)
        self.radCIFAR10.setGeometry(QtCore.QRect(10, 40, 82, 17))
        self.radCIFAR10.setObjectName("radCIFAR10")
        self.radMNIST = QtWidgets.QRadioButton(self.grpDataSet)
        self.radMNIST.setGeometry(QtCore.QRect(10, 17, 91, 20))
        self.radMNIST.setObjectName("radMNIST")
        self.radPrimaHeadPose = QtWidgets.QRadioButton(self.grpDataSet)
        self.radPrimaHeadPose.setGeometry(QtCore.QRect(10, 60, 101, 17))
        self.radPrimaHeadPose.setObjectName("radPrimaHeadPose")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1198, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNEW = QtWidgets.QAction(MainWindow)
        self.actionNEW.setObjectName("actionNEW")
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionNEW)
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabMenuLayer.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CNN ToolKit"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Train Network"))
        self.btnCancelTraining.setText(_translate("MainWindow", "Cancel Training"))
        self.lblNumEpochs.setText(_translate("MainWindow", "Number of Epochs:"))
        self.lblBatchSize.setText(_translate("MainWindow", "Batch Size:"))
        self.lblLearningRate.setText(_translate("MainWindow", "Learning Rate:"))
        self.lblOptimizer.setText(_translate("MainWindow", "Optimizer:"))
        self.cbxOptimizer.setItemText(0, _translate("MainWindow", "Gradient Descent Optimizer"))
        self.cbxOptimizer.setItemText(1, _translate("MainWindow", "Adam Optimizer"))
        self.cbxOptimizer.setItemText(2, _translate("MainWindow", "AdaGrad Optimizer"))
        self.lblTrainingProgress.setText(_translate("MainWindow", "Training Progress"))
        self.lblTestAccuracy.setText(_translate("MainWindow", "Test Set Accuracy:"))
        self.btnTrainNetwork.setText(_translate("MainWindow", "Train Network"))
        self.lblOutputLog.setText(_translate("MainWindow", "Output Log"))
        self.grpNetworkModel.setTitle(_translate("MainWindow", "Define Network Model"))
        self.cbxLayerSelect.setItemText(0, _translate("MainWindow", "Convolutional Layer"))
        self.cbxLayerSelect.setItemText(1, _translate("MainWindow", "Max Pooling Layer"))
        self.cbxLayerSelect.setItemText(2, _translate("MainWindow", "Dropout"))
        self.cbxLayerSelect.setItemText(3, _translate("MainWindow", "Normalizing Layer"))
        self.cbxLayerSelect.setItemText(4, _translate("MainWindow", "Fully Connected Layer"))
        self.cbxLayerSelect.setItemText(5, _translate("MainWindow", "Output"))
        self.btnAddLayer.setText(_translate("MainWindow", "Add New Layer"))
        self.cbxConvPadding.setItemText(0, _translate("MainWindow", "SAME"))
        self.cbxConvPadding.setItemText(1, _translate("MainWindow", "VALID"))
        self.lblConvStDev.setText(_translate("MainWindow", "Std Dev of Weights:"))
        self.cbxConvoWeightInit.setItemText(0, _translate("MainWindow", "Truncated Normal"))
        self.lblConvWeightInit.setText(_translate("MainWindow", "Weight Initialization:"))
        self.lblConvKernelSize.setText(_translate("MainWindow", "Conv Kernel Size:"))
        self.lblConvOutputFilters.setText(_translate("MainWindow", "Num Output Filters:"))
        self.lblConvStride.setText(_translate("MainWindow", "Conv Stride:"))
        self.lblConvPadding.setText(_translate("MainWindow", "Padding:"))
        self.lblConvActFunction.setText(_translate("MainWindow", "Activation Function:"))
        self.cbxConvActFunction.setItemText(0, _translate("MainWindow", "Rectified Linear"))
        self.cbxConvActFunction.setItemText(1, _translate("MainWindow", "Sigmoid"))
        self.btnAddConvLayer.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabConvLayer), _translate("MainWindow", "Conv Layer"))
        self.lblPoolStride.setText(_translate("MainWindow", "Pooling Stride:"))
        self.lblPoolPadding.setText(_translate("MainWindow", "Padding:"))
        self.lblPoolKernelSize.setText(_translate("MainWindow", "Pooling Kernel Size:"))
        self.btnAddMaxPool.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabMaxPooling), _translate("MainWindow", "Max Pooling"))
        self.lblPoolKeepRate.setText(_translate("MainWindow", "Keep Rate:"))
        self.btnAddDropout.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabDropout), _translate("MainWindow", "Dropout"))
        self.lblNormBias.setText(_translate("MainWindow", "Bias:"))
        self.lblNormAlpha.setText(_translate("MainWindow", "Alpha:"))
        self.lblNormBeta.setText(_translate("MainWindow", "Beta:"))
        self.btnAddNormalizing.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabNormalizing), _translate("MainWindow", "Normalizing"))
        self.lblFCWeightInit.setText(_translate("MainWindow", "Weight Initialization:"))
        self.lblFCNumOutputNodes.setText(_translate("MainWindow", "Num Output Nodes:"))
        self.lblFCStDev.setText(_translate("MainWindow", "Std Dev of Weights:"))
        self.cbxFCWeightInit.setItemText(0, _translate("MainWindow", "Truncated Normal"))
        self.cbxFCActFunction.setItemText(0, _translate("MainWindow", "Rectified Linear"))
        self.cbxFCActFunction.setItemText(1, _translate("MainWindow", "Sigmoid"))
        self.lblFCActFunction.setText(_translate("MainWindow", "Activation Function:"))
        self.btnAddFullyConn.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabFullyConnected), _translate("MainWindow", "Fully Connected"))
        self.lblOutputStDev.setText(_translate("MainWindow", "Std Dev of Weights:"))
        self.cbxOutputWeightInit.setItemText(0, _translate("MainWindow", "Truncated Normal"))
        self.lblOutputActFunction.setText(_translate("MainWindow", "Activation Function:"))
        self.cbxOutputActFunction.setItemText(0, _translate("MainWindow", "Rectified Linear"))
        self.cbxOutputActFunction.setItemText(1, _translate("MainWindow", "Sigmoid"))
        self.lblOutputWeightInit.setText(_translate("MainWindow", "Weight Initialization:"))
        self.btnAddOutput.setText(_translate("MainWindow", "Add Layer"))
        self.tabMenuLayer.setTabText(self.tabMenuLayer.indexOf(self.tabOutput), _translate("MainWindow", "Output"))
        self.btnValidateNetwork.setText(_translate("MainWindow", "Validate Network"))
        self.grpDataSet.setTitle(_translate("MainWindow", "Data Set"))
        self.radCIFAR10.setText(_translate("MainWindow", "CIFAR-10"))
        self.radMNIST.setText(_translate("MainWindow", "MNIST"))
        self.radPrimaHeadPose.setText(_translate("MainWindow", "Prima Head Pose"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionNEW.setText(_translate("MainWindow", "New"))
        self.actionLoad.setText(_translate("MainWindow", "Load"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

