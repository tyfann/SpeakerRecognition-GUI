import time
from PyQt5.QtGui import QKeyEvent, QTextCursor
from record import Recorder

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                      QWidget, QMessageBox, QLineEdit)
from PyQt5 import uic

import numpy as np
import os, glob, shutil, sys, torch, warnings, importlib, argparse
from demoSpeakerNet import loadWAV, loadPretrain, mkdir, deldir
from scipy.io.wavfile import write
import threading
import datetime



class mainWindow(QMainWindow):

    def __init__(self):
        super(mainWindow, self).__init__()
        uic.loadUi("ui/mainLayout.ui",self)
        # self.ui = uic.loadUi("ui/mainLayout.ui")
        # self.ui.show()

    def closeEvent(self,e):
        reply = QMessageBox.question(self, '提示',
                    "是否要关闭所有窗口?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)   # 退出程序
        else:
            e.ignore()



class testWindow(QWidget):
    
    m_singal = pyqtSignal(str)
    def __init__(self):
        super(testWindow, self).__init__()
        uic.loadUi("ui/testLayout.ui",self)
        # self.ui = uic.loadUi("ui/testLayout.ui")
        self.recordButton.clicked.connect(self.slot_recordButton)
        self.endButton.clicked.connect(self.slot_endButton)
        self.recordButton.setEnabled(True)
        self.endButton.setEnabled(False)
        self._running = False
        self.textEdit.clear()

    def slot_recordButton(self):
        self._running = True
        self.recordButton.setEnabled(False)
        self.endButton.setEnabled(True)
        threading._start_new_thread(self.__record, ())

    def __record(self):

        threading._start_new_thread(self.loop_record, ("Thread-1", ))
        time.sleep(2)
        threading._start_new_thread(self.loop_record, ("Thread-2", ))

    def showEvent(self, event):
        self.lineEdit.clear()
        self.textEdit.clear()

    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮：Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
                    第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        if self._running:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "还在录音!", QMessageBox.Ok)
            event.ignore()
            return
        
        # reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
        #                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        # if reply == QMessageBox.Yes:
        #     event.accept()
        # else:
        #     event.ignore()

    def loop_record(self, threadName):
        
        file_name = [str(x) for x in range(10)]
        count = 0
        while self._running:
            rec = Recorder()
            rec.start()
            # print("current_thread: "+threading.current_thread().getName()+"\nstartTime: "+str(datetime.datetime.now()))
            time.sleep(4)
            rec.stop()
            audio = "rec_files/test/"+threadName+"_"+file_name[count]+"_test.wav"
            rec.save(audio)
            count += 1
            count %= 10
            self.test(audio)

        rec.stop()
        audio = "rec_files/test/"+threadName+"_"+file_name[count]+"_test.wav"
        rec.save(audio)


    def slot_endButton(self):
        # if testQueue.qsize == 0:
        #     messageBox = QMessageBox(self)
        #     messageBox.information(self, "警告", "未开始录音!", QMessageBox.Ok)
        #     return

        self._running = False

        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "结束测试!", QMessageBox.Ok)
        self.recordButton.setEnabled(True)
        self.endButton.setEnabled(False)

    def test(self, audio):
        threading._start_new_thread(self.__validate, (audio, ))
    
    def __validate(self, audio):
        feat_test = model(loadWAV(audio)).detach()
        feat_test = torch.nn.functional.normalize(feat_test, p=2, dim=1)
        max_score = float('-inf')
        max_audio = ''
        # enroll_audios = glob.glob('rec_files/enroll/*.wav')
        enroll_audios = glob.glob('models/data/enroll_embeddings/*.pt')
        
        for i, enroll_audio in enumerate(enroll_audios):
            enroll_embeddings = torch.load(enroll_audios[i])
            score = float(np.round(- torch.nn.functional.pairwise_distance(enroll_embeddings.unsqueeze(-1),
                                                                           feat_test.unsqueeze(-1).transpose(0,
                                                                                                             2)).detach().numpy(),
                                   4))
            if max_score < score:
                max_score = score
                max_audio = enroll_audio.split('/')[-1].split('.')[0]
        if max_score < -1.0:
            self.lineEdit.setText("unknown person")
            self.m_singal.emit(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"   "+"unknown person")
        else:
            self.lineEdit.setText(max_audio.split('_')[0])
            self.m_singal.emit(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"   "+max_audio)
    
    def show_msg(self, msg):
        self.textEdit.moveCursor(QTextCursor.End)
        self.textEdit.append(msg)
        pass


class enrollWindow(QWidget):
    def __init__(self):
        super(enrollWindow, self).__init__()
        uic.loadUi("ui/enrollLayout.ui",self)
        # self.ui = uic.loadUi("ui/enrollLayout.ui")
        self.recordButton.clicked.connect(self.slot_recordButton)
        self.endButton.clicked.connect(self.slot_endButton)
        self.endButton.setEnabled(False)
        self._running = False
        # self.ui.recordButton.clicked.connect(self.slot_recordButton)
        # self.ui.show()

    # 开始录音
    def slot_recordButton(self):
        self.text = self.lineEdit.text()
        if len(self.text) == 0:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "未输入注册录音用户名称!", QMessageBox.Ok)
            return
        self._running = True
        self.rec = Recorder()
        self.rec.start()
        self.recordButton.setEnabled(False)
        self.endButton.setEnabled(True)
    
    # 结束录音
    def slot_endButton(self):
        # if recQueue.qsize == 0:
        #     messageBox = QMessageBox(self)
        #     messageBox.information(self, "警告", "未开始录音!", QMessageBox.Ok)
        #     return
        self._running = False
        self.rec.stop()
        audio = "rec_files/enroll/"+str(self.text)+"_enroll.wav"
        self.rec.save(audio)
        self.train(audio)
        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "已保存录音!", QMessageBox.Ok)
        self.recordButton.setEnabled(True)
        self.endButton.setEnabled(False)
    
    def showEvent(self, event):
        self.lineEdit.clear()

    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮：Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
                    第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        if self._running:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "还在录音!", QMessageBox.Ok)
            event.ignore()
            return
        
        # reply = QMessageBox.question(self, 'Message',"Are you sure to quit?",
        #                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        # if reply == QMessageBox.Yes:
        #     event.accept()
        # else:
        #     event.ignore()

    def train(self, audio):
        threading._start_new_thread(self.__enroll, (audio, ))
    
    def __enroll(self, audio):
        with torch.no_grad():
            feat_enroll = model(loadWAV(audio)).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            # feat_enroll_list.append(feat_enroll)
            # embeddings = torch.cat(feat_enroll_list, dim=0)
            # enroll_embeddings = torch.mean(embeddings, dim=0, keepdim=True)
            # torch.save(enroll_embeddings, os.path.join('models','data', 'enroll_embeddings', 'pre_spk.pt'), _use_new_zipfile_serialization=False)
            torch.save(feat_enroll, os.path.join('models','data', 'enroll_embeddings', str(self.text)+'.pt'), _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SpeakerNetModel = importlib.import_module('models.ResNetSE34V2').__getattribute__('MainModel')
    global model
    
    model = SpeakerNetModel()
    model = loadPretrain(model, 'models/pretrain.model')
    # deldir('rec_files/enroll')
    deldir('rec_files/test')
    mkdir('rec_files/enroll')
    mkdir('rec_files/test')
    mkdir('models/data/enroll_embeddings')

    app = QApplication(sys.argv)
    mainWin = mainWindow()
    enrollWin = enrollWindow()
    testWin = testWindow()
    testWin.m_singal.connect(testWin.show_msg)

    # mainWin.ui.show()
    mainWin.show()
    mainWin.enrollButton.clicked.connect(enrollWin.show)
    mainWin.testButton.clicked.connect(testWin.show)
    sys.exit(app.exec_())
