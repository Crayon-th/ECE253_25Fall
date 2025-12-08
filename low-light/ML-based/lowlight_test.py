import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import time
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

def lowlight(image_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on device: {device}")

    scale_factor = 12
    data_lowlight = Image.open(image_path)
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()

    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[:h, :w, :]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0).to(device)

    DCE_net = model.enhance_net_nopool(scale_factor).to(device)
    DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth', map_location=device))

    start = time.time()
    enhanced_image, _ = DCE_net(data_lowlight)
    end_time = time.time() - start
    print(f"Processing time: {end_time:.4f}s")

    result_path = image_path.replace('test_data','result_Zero_DCE++')
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)

    return end_time

if __name__ == '__main__':
    filePath = 'data/test_data/'
    file_list = os.listdir(filePath)
    sum_time = 0
    with torch.no_grad():
        for folder in file_list:
            for image in glob.glob(filePath + folder + "/*"):
                print(image)
                sum_time += lowlight(image)
    print(f"Total time: {sum_time:.4f} sec")
