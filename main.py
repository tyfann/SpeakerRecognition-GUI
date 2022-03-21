import time
from PyQt5.QtGui import QKeyEvent, QTextCursor
from openvino.inference_engine import IECore

from record import Recorder
import argparse
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QPlainTextEdit,
                             QWidget, QMessageBox, QLineEdit)
from PyQt5 import uic

import numpy as np
import os, glob, shutil, sys, torch, warnings, importlib, argparse
from demoSpeakerNet import loadWAV, loadPretrain, mkdir, deldir, getKey
from audioProcessing import pcm2wav, sgn, calEnergy, calZeroCrossingRate, endPointDetect
from scipy.io.wavfile import write
import threading
from threading import Condition
import datetime
import librosa
import wave
from queue import Queue

score_threshold = -1.1
time_interval = 1
time_duration = 4


def loadAndCut(filename):
    audio, sr = librosa.load(filename, sr=48000, mono=True)
    # top_db：数字> 0 低于参考值的阈值（以分贝为单位）被视为静音
    clips = librosa.effects.split(audio, top_db=20)
    wav_data = []
    for c in clips:
        data = audio[c[0]: c[1]]
        wav_data.extend(data)
    wav_data = np.array(wav_data)
    # print(filename,'  wav time is ', wav_data.shape[0]/48000)
    feat = np.stack([wav_data], axis=0).astype(np.float)
    feat = torch.FloatTensor(feat)

    if feat.shape[1] / sr <= 0.5:
        return None
    return feat


class mainWindow(QMainWindow):

    def __init__(self):
        super(mainWindow, self).__init__()
        uic.loadUi("ui/mainLayout.ui", self)


    def closeEvent(self, e):
        reply = QMessageBox.question(self, '提示',
                                     "是否要关闭所有窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            e.accept()
            sys.exit(0)  # 退出程序
        else:
            e.ignore()


class testWindow(QWidget):
    m_singal = pyqtSignal(str)
    f_signal = pyqtSignal(str)

    def __init__(self):
        super(testWindow, self).__init__()
        uic.loadUi("ui/testLayout.ui", self)
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
        # 这里可以把vote线程当作消费者线程，而其他几个loop的线程可以当作生产者线程

        threading._start_new_thread(self.consume, ("Thread-vote",))
        for i in range(1, 5):
            threading._start_new_thread(self.produce, ("Thread-" + str(i),))

            time.sleep(time_interval)

    def showEvent(self, event):
        self.lineEdit.clear()
        self.textEdit.clear()

    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮: Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
                    第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        if self._running:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "还在录音!", QMessageBox.Ok)
            event.ignore()
            return

    def consume(self, threadName):
        stat = {}
        count = {}

        while self._running:

            if voteQueue.full():
                for i in range(4):
                    voteKey = voteQueue.get()
                    name = voteKey['name']
                    if name not in count:
                        count[name] = 1
                    else:
                        count[name] += 1
                    if name not in stat:
                        stat[name] = voteKey['value']
                    else:
                        stat[name] += voteKey['value']
                    if i != 0:
                        voteQueue.put(voteKey)

                max_key = max(count, key=count.get)
                max_count = count[max_key]
                result = getKey(count, max_count)
                if len(result) == 1:
                    self.lineEdit.setText(max_key)
                    self.f_signal.emit(
                        str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "   " + max_key + "  vote result.")
                else:
                    min_val = float('-inf')
                    min_name = ''
                    for key in result:
                        if stat[key] > min_val:
                            min_name = key
                            min_val = stat[key]
                    self.lineEdit.setText(min_name)
                    self.f_signal.emit(str(datetime.datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S')) + "   " + min_name + "  vote result.")
                stat = {}
                count = {}

    def produce(self, threadName):
        max_range = 5
        file_name = [str(x) for x in range(max_range)]
        count = 0
        while self._running:
            rec = Recorder()
            rec.start()
            # print("current_thread: "+threading.current_thread().getName()+"\nstartTime: "+str(datetime.datetime.now()))
            time.sleep(time_duration)
            rec.stop()
            if self._running == False:
                break
            audio = "rec_files/test/" + threadName + "_" + file_name[count] + "_test.wav"
            rec.save(audio)

            threading._start_new_thread(self.silence_remove_and_test, (audio,))
            # self.silence_remove_and_test(audio)
            count += 1
            count %= max_range

        rec.stop()
        audio = "rec_files/test/" + threadName + "_" + file_name[count] + "_test.wav"
        rec.save(audio)
        # self.silence_remove(audio)

    def silence_remove_and_test(self, audio):
        f = wave.open(audio, "rb")
        # getparams() 一次性返回所有的WAV文件的格式信息
        params = f.getparams()
        # nframes 采样点数目
        nchannels, sampwidth, framerate, nframes = params[:4]
        # readframes() 按照采样点读取数据
        str_data = f.readframes(nframes)  # str_data 是二进制字符串

        # 以上可以直接写成 str_data = f.readframes(f.getnframes())
        # 转成二字节数组形式（每个采样点占两个字节）
        wave_data = np.fromstring(str_data, dtype=np.short)
        f.close()
        energy = calEnergy(wave_data)

        zeroCrossingRate = calZeroCrossingRate(wave_data)

        sum = 0
        for k in energy:
            sum += k
        aver = sum / len(energy)
        # print(aver)
        if aver < 100000:
            N = []
        else:
            N = endPointDetect(wave_data, energy, zeroCrossingRate)

        # N = endPointDetect(wave_data, energy, zeroCrossingRate)
        # 输出为 pcm 格式
        pcm_path = "rec_files/endpoint/" + audio.split('/')[-1].split('.')[0] + ".pcm"
        with open(pcm_path, "wb") as f:
            i = 0
            while i < len(N) and i + 1 < len(N):
                for num in wave_data[N[i] * 256: N[i + 1] * 256]:
                    f.write(num)
                i = i + 2
        pcm2wav(pcm_path)

        self.__validate(audio)

    def slot_endButton(self):

        self._running = False

        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "结束测试!", QMessageBox.Ok)
        self.recordButton.setEnabled(True)
        self.endButton.setEnabled(False)

    def test(self, audio):
        threading._start_new_thread(self.__validate, (audio,))


    def __validate(self, audio):
        with torch.no_grad():
            feat = loadWAV(audio)
            score_dict = {}
            if not feat.numel():
                score_dict['name'] = "silence"
                score_dict['value'] = -1.0
                self.m_singal.emit(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "   " + "silence type1")
                voteQueue.put(score_dict)
                return
            feat_test = torch.from_numpy(exec_net.infer(inputs={input_blob: [feamodel(feat).numpy()]})[out_blob])
            feat_test = torch.nn.functional.normalize(feat_test, p=2, dim=1)

            max_score = float('-inf')
            max_audio = ''
            for i, enroll_audio in enumerate(enroll_audios):

                score = float(np.round(- torch.nn.functional.pairwise_distance(feat_enroll_list[i].unsqueeze(-1),
                                                                               feat_test.unsqueeze(-1).transpose(0,
                                                                                                                 2)).detach().numpy(),
                                       4))
                if max_score < score:
                    max_score = score
                    max_audio = enroll_audio.split('/')[-1].split('.')[0]

            print('max_score is ', max_score)
            if max_score < score_threshold:
                score_dict['name'] = "silence"
                score_dict['value'] = max_score
                self.m_singal.emit(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "   " + "silence type2")
            else:
                score_dict['name'] = max_audio.split('_')[0]
                score_dict['value'] = max_score
                self.m_singal.emit(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + "   " + max_audio)
            voteQueue.put(score_dict)


    def show_msg(self, msg):
        self.textEdit.moveCursor(QTextCursor.End)
        self.textEdit.append(msg)
        pass

    def show_final_msg(self, msg):
        self.textEdit_Final.moveCursor(QTextCursor.End)
        self.textEdit_Final.append(msg)
        pass


class enrollWindow(QWidget):
    def __init__(self):
        super(enrollWindow, self).__init__()
        uic.loadUi("ui/enrollLayout.ui", self)
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
        audio = "rec_files/enroll/" + str(self.text) + "_enroll.wav"
        self.rec.save(audio)
        self.train(audio)
        messageBox = QMessageBox(self)
        messageBox.information(self, "成功", "已保存录音!", QMessageBox.Ok)
        self.recordButton.setEnabled(True)
        self.endButton.setEnabled(False)

    def showEvent(self, event):
        self.lineEdit.clear()

    def closeEvent(self, event):
        """我们创建了一个消息框，上面有俩按钮: Yes和No.第一个字符串显示在消息框的标题栏，第二个字符串显示在对话框，
                    第三个参数是消息框的俩按钮，最后一个参数是默认按钮，这个按钮是默认选中的。返回值在变量reply里。"""

        if self._running:
            messageBox = QMessageBox(self)
            messageBox.information(self, "警告", "还在录音!", QMessageBox.Ok)
            event.ignore()
            return


    def train(self, audio):
        threading._start_new_thread(self.__enroll, (audio,))

    def __enroll(self, audio):
        with torch.no_grad():
            # feat = loadAndCut(audio)
            feat = loadWAV(audio)

            feat_enroll = torch.from_numpy(exec_net.infer(feamodel(feat))[out_blob])
            feat_enroll = torch.nn.functional.normalize(feat_enroll, p=2, dim=1)
            feat_enroll_list.append(feat_enroll)
            # embeddings = torch.cat(feat_enroll_list, dim=0)
            # enroll_embeddings = torch.mean(embeddings, dim=0, keepdim=True)
            # torch.save(enroll_embeddings, os.path.join('models','data', 'enroll_embeddings', 'pre_spk.pt'), _use_new_zipfile_serialization=False)
            save_path = os.path.join('models', 'data', 'enroll_embeddings', str(self.text) + '.pt')
            torch.save(feat_enroll, save_path, _use_new_zipfile_serialization=False)
            enroll_audios.append(save_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    global feat_enroll_list
    global enroll_audios
    global voteQueue
    global exec_net
    global input_blob, out_blob
    global feamodel

    voteQueue = Queue(4)
    feat_enroll_list = []
    enroll_audios = glob.glob('models/data/enroll_embeddings/*.pt')

    fea = importlib.import_module('models.feature').__getattribute__('MainModel')
    feamodel = fea()

    for enroll_audio in enroll_audios:
        feat_enroll_list.append(torch.load(enroll_audio))


    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='./models/pretrain1', help='enter the model path with model name.')
    args = parser.parse_args()

    model_path = args.model

    model_xml = model_path + ".xml"
    model_bin = model_path + ".bin"

    ie = IECore()

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, device_name="CPU")

    deldir('rec_files/test')
    mkdir('rec_files/enroll')
    mkdir('rec_files/test')
    mkdir('models/data/enroll_embeddings')
    deldir('rec_files/endpoint')
    mkdir('rec_files/endpoint')

    app = QApplication(sys.argv)
    mainWin = mainWindow()
    enrollWin = enrollWindow()
    testWin = testWindow()
    testWin.m_singal.connect(testWin.show_msg)
    testWin.f_signal.connect(testWin.show_final_msg)

    mainWin.show()
    mainWin.enrollButton.clicked.connect(enrollWin.show)
    mainWin.testButton.clicked.connect(testWin.show)
    sys.exit(app.exec_())
