import torch 
from PIL import Image
import tqdm 
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import save_raw_16bit
from zoedepth.utils.misc import colorize
import shutil 

import os
import glob 
import cv2 


# ZoeD_N
conf = get_config("zoedepth", "infer", config_version="kitti")
print(conf)
#conf['save_dir'] = '/home/spurs/.cache/torch/hub/checkpoints'
conf['pretrained_resource'] = 'local::./ZoeD_M12_K.pt'
print(conf)

model_zoe_k = build_model(conf)
#model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)


#model_zoe_k = torch.hub.load(".", "ZoeD_K", source="local", pretrained=True)

#conf = get_config("zoedepth_nk", "infer", config_version='kitti')
#conf['pretrained_resource'] = 'local::./ZoeD_M12_NK.pt'
#model_zoe_nk = build_model(conf)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)

#zoe = model_zoe_nk.to(DEVICE)

img_dir = '/home/spurs/dataset/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/*.png'
des_dir = 'output'
if os.path.exists(des_dir):
    shutil.rmtree(des_dir)

os.makedirs(des_dir, exist_ok=True)

print('zoe.device=', zoe.device)

img_paths = glob.glob(img_dir)
img_paths.sort()

for img in tqdm.tqdm(img_paths):
    image = Image.open(img).convert("RGB") #.to(DEVICE)  # load
    #depth_numpy = zoe.infer_pil(image)  # as numpy

    depth = zoe.infer_pil(image, output_type="tensor", pad_input=False).cpu().detach().numpy()  # as 16-bit PIL Image
    name = os.path.basename(img)
    name = os.path.join(des_dir, name)
    save_raw_16bit(depth, name)

    continue
    
    print(f"depth.min={depth.min()}, depth.max={depth.max()}")
    colored = colorize(depth)

    # save colored output
    fpath_colored = "output_colored.png"
    Image.fromarray(colored).save(fpath_colored)
    os.system(f"cp {img} ./input.png")
    break 

    #depth_tensor = zoe.infer_pil(image, output_type="tensor").cpu().detach().numpy()  # as torch tensor
    

    
    #print(depth_tensor.shape)
    #print(depth_tensor.dtype, depth_tensor.min(), depth_tensor.max())

    #fpath = "output.png"
    #fpath = os.path.join('output')
    #os.makedirs(fpath, exist_ok=True)
    #fpath = os.path.join(fpath, os.path.basename(img))

    #np.save(fpath + '.npy',  depth_tensor)

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
