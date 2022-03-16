import cv2
import math
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

mp_face_detection = mp.solutions.face_detection
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

vid_capture = cv2.VideoCapture('project.avi')

#Variables auxiliares
i=0
detection_counter=0

#Check & metadata
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
else:
  fps = int(vid_capture.get(5))
  print("Frame Rate : ",fps,"frames per second") 
  frame_count = vid_capture.get(7)
  print("Frame count : ", frame_count)
  frame_width = int(vid_capture.get(3))
  print("Width : ",frame_width) 
  frame_height = int(vid_capture.get(4))
  print("height : ",frame_height) 
  frame_size = (frame_width,frame_height)

#inicialización de escritura de video
#vid_writer = cv2.VideoWriter('after_media_pipe.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

#filename: pathname for the output video file
#apiPreference:  API backends identifier -> Para que lo guarde en formato .avi
#fourcc: 4-character code of codec, used to compress the frames (fourcc)
#fps: Frame rate of the created video stream
#frame_size: Size of the video frames
#isColor: If not zero, the encoder will expect and encode color frames.

print("Working...\n")
#Bucle principal
while(vid_capture.isOpened()):
# vid_capture.read() methods returns a tuple, first element is a bool
# and the second is frame
    with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5, model_selection=1) as face_detection:
    #-Minimum confidence value ([0.0, 1.0]) from the face detection model for the 
    #detection to be considered successful. Default to 0.5.
    #-Model_selection: An integer index 0 or 1. Use 0 to select a short-range 
    #model that works best for faces within 2 meters from the camera, and 1 for 
    #a full-range model best for faces within 5 meters.
        ret, frame = vid_capture.read()
        if ret == True:
            resize_and_show(frame)
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i+= 1
            if not results.detections:
                #print("Nº",i," :Nope")
                continue
            else:
                #print("Nº",i," :Yep")
                #print(results.detections)
                annotated_image = frame.copy()
                for detection in results.detections:
                    mp_drawing.draw_detection(annotated_image, detection)
                #print images (tarda demasiado)
                #plt.imshow(annotated_image)
                #plt.show(block=False)
                #plt.pause(0.2)
                #vid_writer.write(annotated_image)
                detection_counter+= 1
                with open('labels_media_pipe_video_original.txt', 'a') as f:
                    f.write('Frame:'+str(i)+'\n')
                    f.write(str(results.detections))
        else:
            print("No hay mas frames o hay error en la lectura, Frame Nº:",i)
            print("Número de frames en el video : ", frame_count)
            break
print("done")




        
