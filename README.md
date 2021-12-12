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

### 需要添加的功能

1、测试文件数量超过一定数量需要删除之前的，保证存储空间不会爆炸（也可以指定产生文件的名称，这样每次就在循环名称存储文件）[已完成]

2、声音score低于1的时候则显示unknown

3、截取的声音若低于一定阈值则忽略这部分声音

### 实验时长记录

1. 测试录音时，每一个loadWAV花费的时间平均为1s左右
2. 比较录音与之前注册的录音所需要花费的时间为0.005s左右


#### 测试语句

请说出：
“你好，这里是（你的名字）正在录音，完毕”