import time
from PyQt5.QtGui import QKeyEvent
from record import Recorder

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                      QWidget, QMessageBox, QLineEdit)
from PyQt5 import uic
import numpy as np
import os, glob, shutil, sys, torch, warnings, importlib, argparse
from demoSpeakerNet import loadWAV, loadPretrain, mkdir, deldir
from scipy.io.wavfile import write
import threading
import datetime

feat_enroll_list = []


class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/mainLayout.ui")


class testWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/testLayout.ui")
        self.ui.recordButton.clicked.connect(self.slot_recordButton)
        self.ui.endButton.clicked.connect(self.slot_endButton)
        self.ui.recordButton.setEnabled(True)
        self.ui.endButton.setEnabled(False)

    def slot_recordButton(self):
        self._running = True
        self.ui.recordButton.setEnabled(False)
        self.ui.endButton.setEnabled(True)
        threading._start_new_thread(self.__record, ())

    def __record(self):

        threading._start_new_thread(self.loop_record, ())
        time.sleep(2)
        threading._start_new_thread(self.loop_record, ())


    def loop_record(self):
        rec = Recorder()
        while self._running:
            rec.start()
            time.sleep(4)
            rec.stop()
            audio = "rec_files/test/"+str(datetime.datetime.now())+"_test.wav"
            rec.save(audio)
            self.test(audio)

        rec.stop()
        audio = "rec_files/test/"+str(datetime.datetime.now())+"_test.wav"
        rec.save(audio)


    def slot_endButton(self):
        # if testQueue.qsize == 0:
        #     messageBox = QMessageBox(self)
        #     messageBox.information(self, "警告", "未开始录音!", QMessageBox.Ok)
        #     return

        self._running = False

        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "结束测试!", QMessageBox.Ok)
        self.ui.recordButton.setEnabled(True)
        self.ui.endButton.setEnabled(False)

    def test(self, audio):
        threading._start_new_thread(self.__validate, (audio, ))
    
    def __validate(self, audio):
        feat_test = model(loadWAV(audio)).detach()
        feat_test = torch.nn.functional.normalize(feat_test, p=2, dim=1)
        max_score = float('-inf')
        max_audio = ''
        enroll_audios = glob.glob('rec_files/enroll/*.wav')
        for i, enroll_audio in enumerate(enroll_audios):
            score = float(np.round(- torch.nn.functional.pairwise_distance(feat_enroll_list[i].unsqueeze(-1),
                                                                           feat_test.unsqueeze(-1).transpose(0,
                                                                                                             2)).detach().numpy(),
                                   4))
            if max_score < score:
                max_score = score
                max_audio = enroll_audio.split('/')[-1]
        self.ui.lineEdit.setText(max_audio.split('_')[0])


class enrollWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/enrollLayout.ui")
        self.ui.recordButton.clicked.connect(self.slot_recordButton)
        self.ui.endButton.clicked.connect(self.slot_endButton)
        self.ui.endButton.setEnabled(False)
        # self.ui.recordButton.clicked.connect(self.slot_recordButton)
        # self.ui.show()

    # 开始录音
    def slot_recordButton(self):
        self.text = self.ui.lineEdit.text()
        if len(self.text) == 0:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "未输入注册录音用户名称!", QMessageBox.Ok)
            return

        self.rec = Recorder()
        self.rec.start()
        self.ui.recordButton.setEnabled(False)
        self.ui.endButton.setEnabled(True)
    
    # 结束录音
    def slot_endButton(self):
        # if recQueue.qsize == 0:
        #     messageBox = QMessageBox(self)
        #     messageBox.information(self, "警告", "未开始录音!", QMessageBox.Ok)
        #     return
        self.rec.stop()
        audio = "rec_files/enroll/"+str(self.text)+"_enroll.wav"
        self.rec.save(audio)
        self.train(audio)
        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "已保存录音!", QMessageBox.Ok)
        self.ui.recordButton.setEnabled(True)
        self.ui.endButton.setEnabled(False)
    
    def train(self, audio):
        threading._start_new_thread(self.__enroll, (audio, ))
    
    def __enroll(self, audio):
        with torch.no_grad():
            feat_enroll = model(loadWAV(audio)).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            feat_enroll_list.append(feat_enroll)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SpeakerNetModel = importlib.import_module('models.ResNetSE34V2').__getattribute__('MainModel')
    global model
    model = SpeakerNetModel()
    model = loadPretrain(model, 'models/pretrain.model')
    deldir('rec_files/enroll')
    deldir('rec_files/test')
    mkdir('rec_files/enroll')
    mkdir('rec_files/test')

    app = QApplication([])
    mainWin = mainWindow()
    enrollWin = enrollWindow()
    testWin = testWindow()

    mainWin.ui.show()
    mainWin.ui.enrollButton.clicked.connect(enrollWin.ui.show)
    mainWin.ui.testButton.clicked.connect(testWin.ui.show)
    sys.exit(app.exec_())
