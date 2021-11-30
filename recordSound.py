# -*- coding: utf-8 -*
import sounddevice as sd
import threading
import os, glob, pandas, numpy, shutil, sys, soundfile
from scipy.io.wavfile import write

"""
  0 Background Music, Core Audio (2 in, 2 out)
  1 Background Music (UI Sounds), Core Audio (2 in, 2 out)
> 2 MacBook Pro麦克风, Core Audio (1 in, 0 out)
< 3 MacBook Pro扬声器, Core Audio (0 in, 2 out)
  4 Filmage Audio Device, Core Audio (2 in, 2 out)
  5 MJAudioRecorder, Core Audio (2 in, 2 out)
  6 聚集设备, Core Audio (1 in, 2 out)
  7 Filmage 聚集设备, Core Audio (3 in, 2 out)
"""

def record_sound():
  fs = 44100 # 采样率44100/48000帧
  sd.default.samplerate = fs
  sd.default.channels = 1
  duration = int(input("Enter the time duration in second: ")) # 持续时间
  myrecording = sd.rec(int(duration * fs)) # 录制音频
  sd.wait() # 阻塞
  write("rec_files/out.wav",fs,myrecording)

t = threading.Thread(target=record_sound)
t.start()

