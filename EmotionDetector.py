import cv2 
from time import sleep 
from PIL import Image
import numpy as np
#import pandas as pd
#from EmoPy.src.fermodel import FERModel
import threading
from EmoPy1.EmoPy.src.fermodel import FERModel

def Capturing_video():
  counter = 0
  CapturedVideo = cv2.VideoCapture('ObamaVideo.mp4')
  FrameRate = CapturedVideo.get(5)
  success = True
  while success:
    frameId = CapturedVideo.get(1)
    success, image = CapturedVideo.read()
    if (frameId % np.floor(FrameRate) == 0):
        im.append(image)
        counter += 1


def Predicting(im, index):
  cv2.imwrite("Photo.jpg", im)
  print('Predicting emotion of Frame %d...' % (index+1))
  index += 1
  p, acc = model.predict("Photo.jpg")
  print(f'Dominant: {p}, acc: {acc*100}')
  return p, acc


if __name__ =='__main__':


    target_emotions = ['calm', 'anger', 'happiness']
    target_emotions2 = ['surprise', 'disgust', 'happiness']

    model = FERModel(target_emotions, verbose=True)
    model2 = FERModel(target_emotions2, verbose=True)

    im = []    
    emo=[]
    percentage=[]
    Capturing_video = threading.Thread(target=Capturing_video)
    Capturing_video.start()
    sleep(2)
    num = 0
    try:
        while True:
            em,acc=Predicting(im[num], num)
            emo.append(em)
            percentage.append(acc)
            num += 1
            sleep(1)
    except:
        print('==============================================')
        print('==============================================')
        print('==============================================')
        print(emo)
        print(np.mean(percentage))
        print("End of Video")
