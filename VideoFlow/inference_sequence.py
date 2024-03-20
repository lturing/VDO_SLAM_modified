import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from utils import flow_viz

from core.Networks import build_network

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import itertools
import imageio
from tqdm import tqdm 
import cv2

import warnings
warnings.filterwarnings("ignore")


def read_from_flo(file):
    # https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py
    TAG_FLOAT = 202021.25
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)[0]
    h = np.fromfile(f, np.int32, count=1)[0]
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    #print(f"w={w}, h={h}")

    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2)) 
    f.close()

    return flow


def write_as_flo(flow, filename):
    # https://github.com/Johswald/flow-code-python/blob/master/writeFlowFile.py
    TAG_STRING = 'PIEH'.encode('utf-8')
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % filename[-4:]

    height, width, nBands = flow.shape
    assert nBands == 2, "Number of bands = %r != 2" % nBands
    u = flow[: , : , 0]
    v = flow[: , : , 1]
    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    f = open(filename,'wb')
    f.write(TAG_STRING)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)

    f.close()


def prepare_image(seq_dir, input_frames):
    print(f"preparing image...")
    print(f"Input image sequence dir = {seq_dir}")

    images = []

    image_list = sorted(os.listdir(seq_dir))

    for fn in image_list:
        img = Image.open(os.path.join(seq_dir, fn))
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
    
    # padding the first and the last
    images.insert(0, images[0])
    #images.append(images[-1])
    padding = input_frames - 2 - (len(images) % (input_frames - 2))
    last = images[-1]
    for _ in range(padding):
        images.append(last)

    return torch.stack(images)

def vis_pre(flow_pre, vis_dir):

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    N = flow_pre.shape[0]

    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx+2, idx+3))
    
    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(vis_dir, idx-N//2+2, idx-N//2+1))

@torch.no_grad()
def MOF_inference(model, cfg):

    model.eval()

    input_images = prepare_image(cfg.seq_dir, cfg.input_frames)

    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)
    # 
    # range(0, len, 3)
    # [0, 5], [3, 8], [6, 11]
    # [1, 5], [4, 8], [7, 11]


    shape = input_images[0].shape[1:]
    size = (shape[1], shape[0] * 2)
    print(f"video size: {size}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = 'output.mp4'
    fps = 10

    videoWriter = cv2.VideoWriter(output_path, fourcc, fps, size, True)
    
    for idx_image in tqdm(range(0, input_images.shape[0]-cfg.input_frames+1, cfg.input_frames-2)):
        inputs = input_images[idx_image:idx_image+cfg.input_frames]
        inputs = inputs[None].cuda()
        padder = InputPadder(inputs.shape)
        inputs = padder.pad(inputs)
        flow_pre, _ = model(inputs, {})
        flow_pre = padder.unpad(flow_pre[0]).cpu()

        N = flow_pre.shape[0]
        #print(f"N={N}, inputs.shape={inputs.shape}")
        for idx in range(N//2):
            flow_data = flow_pre[idx].permute(1, 2, 0).numpy()
            flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
            image = Image.fromarray(flow_img)
            
            name = "flow_{:06}_to_{:06}".format(idx_image+idx, idx_image+idx+1)

            image.save('{}/{}.png'.format(cfg.vis_dir, name))

            write_as_flo(flow_data, '{}/{}.flo'.format(cfg.vis_dir, name))

            video = np.zeros((shape[0] * 2, shape[1], 3), dtype=np.uint8)

            video[:shape[0], :, :] = flow_img.astype(np.uint8)[:, :, ::-1]
            video[shape[0]:, :, :] = input_images[idx_image+idx+1].permute(1, 2, 0).numpy()[:, :, ::-1].astype(np.uint8)

            #video[:shape[0], :, :] = input_images[idx_image+idx].permute(1, 2, 0).numpy().astype(np.uint8)
            #video[shape[0]:, :, :] = flow_img.astype(np.uint8)
            videoWriter.write(video)
    videoWriter.release()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='MOF')
    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')
    parser.add_argument('--model', default="VideoFlow_ckpt/MOF_kitti.pth", help='the path to the checkpoint')
    
    args = parser.parse_args()

    from configs.multiframes_sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    print(f"cfg={cfg}")
    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))
    
    with torch.no_grad():
        from configs.multiframes_sintel_submission import get_cfg
        flow_pre = MOF_inference(model.module, cfg)





