import cv2
import matplotlib.pyplot as plt
filename = 'data/demo/frames/00002575.jpg'
#00002575.jpg,553,71,752,275
img = cv2.imread(filename)
if img is None:
    print('Could not read image')
#Make copy of the image
imageLine = img.copy()
# Draw the image from point A to B
pointA = (553,71)
pointB = (752,275)
print("pinto")
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3)
cv2.imshow('Image Line', imageLine)
cv2.imwrite('a.jpg',imageLine)