# SpeakerRecognition-GUI

## Installation（安装）

### 基础Audio、torch环境配置：

```shell
pip install -r requirements.txt
```

### Openvino安装（以MacOS系统为例）：

根据openvino官网提供的安装指南：https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_macos.html

进入https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html选择相应配置的openvino软件下载并安装

安装完成后需要将openvino的环境变量添加到当前机器中：

```shell
vim ~/.bash_profile
```

在文件最后一行加入：

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

重启当前terminal即可完成环境配置

### 模型引入

在models文件夹下放入对应的模型，并在main.py中修改下面的代码行：

```python
fea = importlib.import_module('models.feature').__getattribute__('MainModel')
```

## Preprocessing（预处理）

### onnx模型转换为openvino下的IR模型

以下代码可以将.onnx模型转换成openvino格式的模型

```shell
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer

python mo_onnx.py --input_model ~/model/pretrain.onnx --output_dir ~/model
```

处理后会得到如下文件目录：

```shell
.
├── pretrain.bin
├── pretrain.mapping
├── pretrain.onnx
└── pretrain.xml
```

将此文件夹下的4个文件放入到models文件夹下即可

## Running（运行）

### 指定模型运行所使用的预训练模型

用python执行main.py文件即可运行语音识别的GUI程序，如不指明模型名称则默认使用项目提供的pretrain1模型

```shell
python main.py --model ./models/pretrain1
```

## Tuning（调参）

### score_threshold

分数阈值，为负数，默认值为`-1.1`

### time_interval

采样时间间隔，默认值为`1`

### time_duration

单次采样持续时间，默认值为`4`

### voteQueue

投票队列，长度默认值为`4`

## Using（使用）

### 初始界面

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h0hbqf9bjpj20vy0sy0tf.jpg" alt="初始界面" style="zoom:50%;" />

### 新用户注册音频

点击注册录音按钮，进入如下界面：

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h0hbsa0r0rj20ue0pwdgn.jpg" alt="注册页面" style="zoom:50%;" />

点击开始录音即可打开录音功能，之后点击结束录音即可完成当前用户的音频录入。

### 测试录音功能

退回初始页面后，点击测试录音即可进入测试界面：

<img src="https://tva1.sinaimg.cn/large/e6c9d24egy1h0hbu90eayj214s0u0t9x.jpg" alt="测试录音页面" style="zoom:50%;" />

如上图所示，左侧的空白文本框显示每一次音频模型判定的结果，右侧的空白文本框则表示4次连续音频的投票结果

点击开始测试后即进入测试模块，点击结束测试即可完成测试
