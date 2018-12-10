import cv2 
from time import sleep 
import numpy as np
import pandas as pd
import threading
from Emotion.EmoPy.src.fermodel import FERModel

def Capturing_video(video):
  counter = 0
  CapturedVideo = cv2.VideoCapture(video)
  FrameRate = CapturedVideo.get(5)
  success = True
  while success:
    frameId = CapturedVideo.get(1)
    success, image = CapturedVideo.read()
    if (frameId % np.floor(FrameRate) == 0):
        im.append(image)
        counter += 1


def Predicting(model,im, index):
  cv2.imwrite("Photo.jpg", im)
  print('Predicting emotion of Frame %d...' % (index+1))
  index += 1
  list_Emotions=[]
  list_accuracy=[]
  
  for i in range(len(model)):
      emotion, accuracy = model['Model'+str(i)].predict("Photo.jpg")
      list_Emotions.append(emotion)
      list_accuracy.append(accuracy)
      print(f'Dominant: {emotion}, acc: {accuracy*100}, Model {i}')
  return list_Emotions, list_accuracy


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
            happiness, anger """

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
    # Lista de Modelos 
    model=Model()

    im = []    
    emotions=[]
    accuracy=[]
    
    Capturing_video = threading.Thread(target=Capturing_video('ObamaVideo.mp4'))
    Capturing_video.start()
    sleep(2)
    num = 0

    try:
        while True:
            em,acc=Predicting(model, im[num], num)
            emotions.append(em)
            accuracy.append(acc)
            num += 1
            sleep(1)
    except:
        print('==========ANALISIS==TERMINADO=================')
        print('==============================================')
        print('==============================================')
        print('==============================================')
        
        emotions= pd.DataFrame(np.array(emotions).reshape(31*9,-1),columns=['Emotions'])
        accuracy= pd.DataFrame(np.array(accuracy).reshape(31*9,-1),columns=['Accuracy'])
        EmotionsM=pd.concat([emotions, accuracy], axis=1)
        Emotions=EmotionsM.set_index('Emotions')
        
        print('==============================================')
        emo, cantidad=np.unique(EmotionsM.iloc[:,0], return_counts=True)
        print(f'Emotions: {emo}\n Cantidad Detectada : {cantidad}')
        
        for i in emo:
            mean_Emotion = Emotions.loc[i].mean()
            print(f'{i} AVG {mean_Emotion} ' )
        print("End of Video")
