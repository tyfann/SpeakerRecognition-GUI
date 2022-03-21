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



### 代码运行前须知：

需要引入models，models的样例如下链接

[models](https://github.com/TaoRuijie/SpeakerRecognitionDemo/tree/main/models)

引入完models文件到与main.py同目录下即可



## Preprocessing（预处理）

### onnx模型转换为openvino下的IR模型

```shell
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer

python mo_onnx.py --input_model ~/keras-learn/models/resnet50-v2-7.onnx --output_dir ~/keras-learn/models
```



## Running（运行）


用python执行main.py文件即可运行语音识别的GUI程序
```shell
python main.py
```



## Tuning（调参）
