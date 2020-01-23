import torch
from aux import str_to_mask 
import matplotlib.pyplot as plt
from PIL import Image
from model import SomeModel
from load import MyDataset
import numpy as np


SLICE_HEIGHT = 140
SLICE_WIDTH = 140

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SomeModel()
# model.load_state_dict(torch.load('model_dict.pt'))
# model.load_state_dict(torch.load('model_dict.pt', map_location=lambda storage, loc: storage))
model = torch.load('model_epoch_4.pt') 
model = model.to(device)
model.eval()


PREDICT_PATH = "../data/train_images/"
LABEL_PATH = "../data/train.csv"
TRAIN_PARTITION = 0.8

prediction_dataset = MyDataset(train_dir=PREDICT_PATH, labels=LABEL_PATH, train=False, train_partition=TRAIN_PARTITION)
print(device)

for sample, _ in prediction_dataset:
    blank_image = Image.new('RGB', (2100, 1400))
    blank_mask = Image.new('RGB', (2100, 1400))
    sample = sample.transpose(2,0,1)
    max_value = np.max(sample)
    for i in range(10):
        for j in range(15):
            with torch.no_grad():
                row_pix = i * SLICE_HEIGHT
                col_pix = j*SLICE_WIDTH
                aux_img = sample[:, row_pix:row_pix + SLICE_HEIGHT, col_pix:col_pix + SLICE_WIDTH]
                aux_img = torch.Tensor(aux_img).unsqueeze(0)
                aux_img = aux_img.to(device)
                pred = model(torch.Tensor(aux_img))
                pred = pred.squeeze(0)
                pred = pred.permute(1,2,0)
                pred = np.array(pred)
                aux_img = aux_img.squeeze(0)
                aux_img = aux_img.permute(1,2,0)
                aux_img = np.array(aux_img).astype(np.float32)
                aux_img = (aux_img * 255 / max_value).astype('uint8')
                image = Image.fromarray(aux_img)
                pred = pred * 255.
                pred = Image.fromarray(pred[:,:,0])

                blank_image.paste(image,(col_pix, row_pix)) 
                blank_mask.paste(pred,(col_pix, row_pix)) 

    plt.imshow(blank_mask)
    plt.show() 
