from PyQt5.Qt import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                      QWidget, QThread, QMessageBox, QLineEdit, QMutex)
from PyQt5 import uic
import sounddevice as sd
import numpy as np
import pandas as pd
import os, glob, shutil, sys, soundfile, torch, warnings, importlib, argparse, copy
from scipy.io.wavfile import write
import threading
import datetime


def get_index_to_name(input_list):
    output_list = copy.deepcopy(input_list)
    for i in range(len(output_list)):
        output_list[i] = output_list[i].split('/')[-1].split('-')[0]
    return output_list


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def loadWAV(filename):
    audio, sr = soundfile.read(filename)
    if len(audio.shape) == 2:  # dual channel will select the first channel
        audio = audio[:, 0]
    feat = np.stack([audio], axis=0).astype(np.float)
    feat = torch.FloatTensor(feat)
    return feat


def loadPretrain(model, pretrain_model):
    self_state = model.state_dict()
    # if using on cuda
    # loaded_state = torch.load(pretrain_model, map_location={'cuda:1':'cuda:0'})

    # if using on cpu
    loaded_state = torch.load(pretrain_model, map_location="cpu")
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("__S__.", "")
            if name not in self_state:
                continue
        self_state[name].copy_(param)
    self_state = model.state_dict()
    return model


feat_enroll_list = []


class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/mainLayout.ui")
        self.ui.testButton.clicked.connect(self.slot_testButton)
        # self.ui.show()

    def slot_testButton(self):
        my_t = threading.Thread(target=self.testVoice())
        my_t.start()

    def testVoice(self):
        # while True:
        fs = 44100  # 采样率44100/48000帧
        sd.default.samplerate = fs
        sd.default.channels = 1
        # duration = int(input("Enter the time duration in second: ")) # 持续时间
        duration = 4
        myrecording = sd.rec(int(duration * fs))  # 录制音频
        sd.wait()  # 阻塞

        time = datetime.datetime.now()
        self.audio = "rec_files/test/"+str(time)+"_test.wav"
        write(self.audio, fs, myrecording)
        my_t = threading.Thread(target=self.validateVoice())
        my_t.start()

    def validateVoice(self):
        feat_test = model(loadWAV(self.audio)).detach()
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
        self.ui.textEdit.setPlaceholderText(max_audio.split('_')[0])


class enrollWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/enrollLayout.ui")
        self.ui.recordButton.clicked.connect(self.slot_recordButton)
        # self.ui.show()

    def slot_recordButton(self):
        if len(self.ui.lineEdit.text()) == 0:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "未输入注册录音用户名称!", QMessageBox.Ok)
            return
        # self.rt = EnrollThread()
        # self.rt.start()
        my_t = threading.Thread(target=self.recordVoice())
        my_t.start()

    def recordVoice(self):
        enroll_id = self.ui.lineEdit.text()
        fs = 44100  # 采样率44100/48000帧
        sd.default.samplerate = fs
        sd.default.channels = 1
        duration = 2.5
        myrecording = sd.rec(int(duration * fs))  # 录制音频
        sd.wait()  # 阻塞

        audio = "rec_files/enroll/" + str(enroll_id) + "_enroll.wav"
        write(audio, fs, myrecording)
        with torch.no_grad():
            feat_enroll = model(loadWAV(audio)).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            feat_enroll_list.append(feat_enroll)


class RecoThread(QThread):
    def __init__(self):
        super(RecoThread, self).__init__()

    def run(self):
        w = mainWindow()
        w.testVoice()


class EnrollThread(QThread):
    def __init__(self):
        super(EnrollThread, self).__init__()

    def run(self):
        w = enrollWindow()
        w.recordVoice()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    SpeakerNetModel = importlib.import_module('models.ResNetSE34V2').__getattribute__('MainModel')
    global model
    model = SpeakerNetModel()
    model = loadPretrain(model, 'models/pretrain.model')
    mkdir('rec_files/enroll')
    mkdir('rec_files/test')

    app = QApplication([])
    mainWin = mainWindow()
    enrollWin = enrollWindow()

    mainWin.ui.show()
    mainWin.ui.enrollButton.clicked.connect(enrollWin.ui.show)
    sys.exit(app.exec_())
