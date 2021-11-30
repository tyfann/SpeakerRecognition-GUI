# SpeakerRecognition-GUI

### 代码运行前须知：

需要引入models，models的样例如下链接

[models](https://github.com/TaoRuijie/SpeakerRecognitionDemo/tree/main/models)

引入完models文件到与main.py同目录下即可

### 执行代码步骤


用python执行main.py文件即可运行语音识别的GUI程序
```shell
python main.py
```


### 当前已实现功能：

1、注册录音界面与测试录音界面Qt实现

2、能够完成定长（2.5s）的注册录音写入文件夹下，并且训练模型后测试录音，将定长（2.5s）的测试录音写入文件夹下，同时用模型进行预测。