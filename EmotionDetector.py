import cv2 
from time import sleep 
from PIL import Image
import numpy as np
import pandas as pd
import threading
from Emotion.EmoPy.src.fermodel import FERModel

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


def Model():
    """Combinations of emotions:
            fear, anger, calm, surprise
            surprise, happiness, disgust
            fear, anger, surprise
            fear, anger, calm
            happiness, anger, calm
            fear, anger, disgust
            surprise, calm, disgust
            surprise, sadness, disgust
            happiness, ange """

    Emotions = [['calm', 'anger', 'happiness'],
                ['surprise', 'disgust', 'happiness'],
                ['fear', 'anger', 'surprise'],
                ['fear', 'anger', 'calm'],
                ['happiness', 'anger', 'calm'],
                ['fear', 'anger', 'disgust'],
                ['surprise', 'calm', 'disgust'],
                ['surprise', 'sadness', 'disgust'],
                ['happiness', 'anger']]

    Emotions = pd.DataFrame(Emotions)

    models = {}
    for i in range(len(Emotions)):
        target = list(Emotions.iloc[i, :])
        if i < len(Emotions)-1:
            models['Model' + str(i)] = FERModel(target, verbose=True)

        else:
            models['Model' + str(i)] = FERModel(target[:-1], verbose=True)
    return models

if __name__ =='__main__':


    target_emotions = ['calm', 'anger', 'happiness']
    target_emotions2 = ['surprise', 'disgust', 'happiness']

    model = FERModel(target_emotions, verbose=True)
    model2 = FERModel(target_emotions2, verbose=True)

    im = []    
    emotions=[]
    percentage=[]
    Capturing_video = threading.Thread(target=Capturing_video)
    Capturing_video.start()
    sleep(2)
    num = 0
    try:
        while True:
            em,acc=Predicting(im[num], num)
            emotions.append(em)
            percentage.append(acc)
            num += 1
            sleep(1)
    except:
        print('==============================================')
        print('==============================================')
        print('==============================================')
        print('==============================================')
        EmotionsM= pd.DataFrame(np.vstack((emotions,percentage)).T,
                     columns=['Emotions','Accuracy'])
        EmotionsM['Accuracy']=EmotionsM['Accuracy'].astype('float')
        EmotionsM['Emotions']=EmotionsM['Emotions'].astype('category')
        Emotions=EmotionsM.set_index('Emotions')
        
        print('==============================================')
        emo, cantidad=np.unique(EmotionsM.iloc[:,0], return_counts=True)
        print(f'Emotions: {emo}, Cantidad Detectada : {cantidad}')

        for i in emo:
            mean_Emotion = Emotions.loc[i].mean()
            print(f'{i} AVG {mean_Emotion} ' )
        print("End of Video")
