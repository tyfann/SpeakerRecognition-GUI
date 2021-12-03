import time
import pyaudio
import wave
from queue import Queue

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                      QWidget, QMessageBox, QLineEdit)
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

class Recorder(QWidget):
    def __init__(self, chunk=1024, channels=1, rate=44100):
        super().__init__()
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []

    def start(self):
        threading._start_new_thread(self.__recording, ())
    
    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)
        while(self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()
  
    def stop(self):
        self._running = False

    def save(self, filename): 
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        with torch.no_grad():
            feat_enroll = model(loadWAV(filename)).detach()
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            feat_enroll_list.append(feat_enroll)
        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "已保存录音!", QMessageBox.Ok)

class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/mainLayout.ui")
        # self.ui.show()


class testWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.audio = Queue()
        self.ui = uic.loadUi("ui/testLayout.ui")
        self.ui.recordButton.clicked.connect(self.slot_testButton)
        self.ui.endButton.clicked.connect(self.slot_endButton)
        # self.ui.show()

    def slot_testButton(self):

        my_rt1 = threading.Thread(target=self.recordVoice())
        my_rt2 = threading.Thread(target=self.recordVoice())
        my_vt = threading.Thread(target=self.validateVoice())
        my_rt1.start()
        my_vt.start()
        time.sleep(2)
        my_rt2.start()

    def slot_endButton(self):
        # 终止录音
        print("stopped")

    def recordVoice(self):
        while True:
            fs = 44100  # 采样率44100/48000帧
            sd.default.samplerate = fs
            sd.default.channels = 1
            # duration = int(input("Enter the time duration in second: ")) # 持续时间
            duration = 4
            myrecording = sd.rec(int(duration * fs))  # 录制音频
            sd.wait()  # 阻塞

            time = datetime.datetime.now()
            audio = "rec_files/test/" + str(time) + "_test.wav"
            write(audio, fs, myrecording)
            self.audio.put(audio)

    def validateVoice(self):
        feat_test = model(loadWAV(self.audio.get())).detach()
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

recQueue = Queue()

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

        rec = Recorder()
        recQueue.put(rec)
        rec.start()
        self.ui.recordButton.setEnabled(False)
        self.ui.endButton.setEnabled(True)
    
    # 结束录音
    def slot_endButton(self):
        if recQueue.qsize == 0:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "未开始录音!", QMessageBox.Ok)
            return
        rec = recQueue.get()
        rec.stop()
        rec.save("rec_files/enroll/"+str(self.text)+"_enroll")
        self.ui.recordButton.setEnabled(True)
        self.ui.endButton.setEnabled(False)


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
    testWin = testWindow()

    mainWin.ui.show()
    mainWin.ui.enrollButton.clicked.connect(enrollWin.ui.show)
    mainWin.ui.testButton.clicked.connect(testWin.ui.show)
    sys.exit(app.exec_())
