import os
import cv2
import torch
import numpy as np
import torchvision
from torch.autograd import Variable
from train_config import config as cfg
from lib.core.base_trainer.DeAP import DeAP

def main():
    blur_path = cfg.TEST.blur
    out_path = cfg.TEST.restored
    if not os.access(out_path, os.F_LOCK):
        os.mkdir(out_path)
    model = DeAP().cuda()
    model.load_state_dict(torch.load('./checkpoint/0.pth'))
    model = model.eval()

    iteration = 0
    for file in os.listdir(blur_path):
        if not os.path.isdir(out_path + '/' + file):
            os.mkdir(out_path + '/' + file)
        for img_name in os.listdir(blur_path + '/' + file):
            img = cv2.imread(blur_path + '/' + file + '/' + img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img / 255, (2, 0, 1)).astype('float32')) - 0.5
            with torch.no_grad():
                iteration += 1
                img_tensor = Variable(img_tensor.unsqueeze(0)).cuda()
                result_image = model(img_tensor, img_tensor)
                result_image = result_image + 0.5
                out_file_name = out_path + '/' + file + '/' + img_name
                torchvision.utils.save_image(result_image, out_file_name)
                print(iteration)

if __name__ == '__main__':
    main()