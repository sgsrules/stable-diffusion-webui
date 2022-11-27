import os
import cv2
import torch
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

img = ['000001.png', '000002.png']
exp = 4
ratio = 0
rthreshold = 0.02
rmaxcycles = 8
modelDir='train_log'


from train_log.RIFE_HDv3 import Model
model = Model()
model.load_model(modelDir, -1)
print("Loaded v3.x HD model.")
if not hasattr(model, 'version'):
    model.version = 0
model.eval()
model.device()
print(model.version)

img0 = cv2.imread(img[0], cv2.IMREAD_UNCHANGED)
img1 = cv2.imread(img[1], cv2.IMREAD_UNCHANGED)
img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
n, c, h, w = img0.shape
ph = ((h - 1) // 64 + 1) * 64
pw = ((w - 1) // 64 + 1) * 64
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)
if ratio:
    img_list = [img0, model.inference(img0, img1, ratio), img1]
else:
    img_list = [img0]
    for i in range(exp):
        img_list.append(model.inference(img0, img1, (i+1) / (exp+1)))
    img_list.append(img1)

if not os.path.exists('output'):
    os.mkdir('output')
for i in range(len(img_list)):
    cv2.imwrite('output/{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
