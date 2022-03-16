import cv2
import numpy as np
import glob

Numero=len(glob.glob("/home/alvaro.nieva/Documents/TFM-MUIT/attention-target-detection-master_Ibra/data/demo/frames/*.jpg"))
print("Número de imágenes en la carpeta:",Numero)
fps = 2
 
img_array = []
for filename in glob.glob('/home/alvaro.nieva/Documents/TFM-MUIT/attention-target-detection-master_Ibra/data/demo/frames/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print("Done")

