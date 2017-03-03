import sys
import CIFAR10 as cifar
import UIDesign as design
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QThread, pyqtSignal


class workerThread(QThread):

    def __init__(self, num_epochs, batch_size, learning_rate, optimizer):
        QThread.__init__(self)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.epochProgress = pyqtSignal(int)
        testSetAccuracy = pyqtSignal(float)


    def run(self):
        accuracy = cifar.train_network(self.num_epochs, self.batch_size, self.learning_rate, self.optimizer)
        self.testSetAccuracy.emit(accuracy)


    def __del__(self):
        self.wait()




class CNNApp(QMainWindow, design.Ui_MainWindow):

    def __init__(self):  
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.btnTrainNetwork.clicked.connect(self.trainButtonClicked)


    def trainButtonClicked(self):
        try:
            '''
            num_epochs = int(self.txtNumEpochs.text())
            batch_size = int(self.txtBatchSize. text())
            learning_rate = float(self.txtLearningRate.text())
            optimizer = int(self.cbxOptimizer.currentIndex())
            '''
            num_epochs = 400
            batch_size = 64
            learning_rate = 0.05
            optimizer = 1
        except ValueError:
            print("Number of Epochs, Batch Size and Learning Rate must be a number!")
        else:
            #self.get_thread = workerThread()
            thread = workerThread(num_epochs, batch_size, learning_rate, optimizer)
            thread.start()
            #self.txtTestAccuracy.setText(str(accuracy))



def main():
    app = QApplication(sys.argv)
    ui = CNNApp()
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    