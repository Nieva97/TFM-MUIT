import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
#from scipy.misc import imresize #esta deprecated
from skimage.transform import resize

from model import ModelSpatial
from utils import imutils, evaluation
from config import *

#python3 demo.py --vis_mode arrow
parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
parser.add_argument('--image_dir', type=str, help='images', default='data/demo/frames')
parser.add_argument('--head', type=str, help='head bounding boxes', default='data/demo/person1.txt')
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
args = parser.parse_args()


def _get_transform():# no entiendo de donde saca input_*, no lo define en ningun sitio
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)#los numeros de arriba??


#df establece los limites de la bounding-box de la cara
def run():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)    #lee la matriz sin el indice
    df['left'] -= (df['right']-df['left'])*0.1                      #y pone nombre a las columnas
    df['right'] += (df['right']-df['left'])*0.1
    df['top'] -= (df['bottom']-df['top'])*0.1
    df['bottom'] += (df['bottom']-df['top'])*0.1

    #persona1.txt:
    ##00002575.jpg,553,71,752,275
    #Corresponde con (553,71) esquina superior izquierda, (752,275) esquina inferior derecha

    # set up data transformation
    test_transforms = _get_transform()  #TENGO QUE MIRAR QUE HACE ESTO

    #Ver que hace todo el model.py
    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    #variable auxiliar
    x=0 #no uso i porque empieza en 25575

    with torch.no_grad():   #desahbilita el calculo de gradiente -> Reduce memoria utilizada
                            #XQ se usa? porque ya se ha entrenado el modelo y solo se quiere lanzar
                            #el modelo a traves de la NN
        for i in df.index:
            #Lee la imagen y saca sus medidas
            frame_raw = Image.open(os.path.join(args.image_dir, i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]
            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0).cuda()#no entiendo que hacen estas 3 lineas
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head) 

            # heatmap modulation                                #¿esto para que sirve?
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            #norm_map = imresize(raw_hm, (height, width)) - inout
            #imresize esta deprecated
        
            norm_map = resize(raw_hm, (height, width)) - inout
            # vis
            plt.close()
            fig = plt.figure()
            fig.canvas.manager.window.move(0,0)
            plt.axis('off')
            plt.imshow(frame_raw)   #aqui te muestra la imagen sin pintar nada

            ax = plt.gca()
            rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
            ax.add_patch(rect)

            if args.vis_mode == 'arrow':
                if inout < args.out_threshold: # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                    circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                    ax.add_patch(circ)
                    plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))
            else:
                plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

            plt.show(block=False)
            #guardamos la imagen
            x = x+1
            name="resultados/figura_"+str(x)    #si le pones png te lo pone 2 veces
            print(name)
            plt.savefig(name) #save as jpg
            plt.pause(0.2)

        print('DONE!')


if __name__ == "__main__":
    run()
