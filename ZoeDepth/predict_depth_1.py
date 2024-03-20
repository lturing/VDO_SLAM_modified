import torch 
from PIL import Image
import tqdm 
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize

import os
import glob 
import cv2 


# ZoeD_N
conf = get_config("zoedepth", "infer", config_version='kitti')
print(conf)
#conf['save_dir'] = '/home/spurs/.cache/torch/hub/checkpoints'
conf['pretrained_resource'] = 'local::./ZoeD_M12_K.pt'
model_zoe_n = build_model(conf)
#model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

img_dir = '/home/spurs/dataset/30fps/2023_02_21_14_04_08/data/img_small/*.png'
img_dir = '/home/spurs/dataset/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/*.png'
for img in tqdm.tqdm(glob.glob(img_dir)):
    image = Image.open(img).convert("RGB")  # load
    #depth_numpy = zoe.infer_pil(image)  # as numpy

    #depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
    depth_tensor = zoe.infer_pil(image, output_type="tensor").cpu().detach().numpy()  # as torch tensor

    
    #print(depth_tensor.shape)
    #print(depth_tensor.dtype, depth_tensor.min(), depth_tensor.max())

    fpath = "output.png"
    fpath = os.path.join('output')
    os.makedirs(fpath, exist_ok=True)
    fpath = os.path.join(fpath, os.path.basename(img))

    np.save(fpath + '.npy',  depth_tensor)
    break 

    ''' 
    save_raw_16bit(depth_tensor, fpath)

    image = Image.open(img).convert("L") 
    image = np.asarray(image)

    print(image.shape, image.min(), image.max())

    #cv2.imwrite(fpath, depth_tensor)
    #colored = colorize(depth_tensor)
    #colored = colorize(depth_tensor, 0, 10)
    #Image.fromarray(colored).save(fpath)

    '''
