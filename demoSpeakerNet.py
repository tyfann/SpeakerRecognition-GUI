import os, soundfile, torch, copy, shutil
import numpy as np

def deldir(path):
    folder = os.path.exists(path)
    if folder:             
        shutil.rmtree(path)  

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_index_to_name(input_list):
    output_list = copy.deepcopy(input_list)
    for i in range(len(output_list)):
        output_list[i] = output_list[i].split('/')[-1].split('-')[0]
    return output_list


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