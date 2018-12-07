import numpy as np
from pandas import DataFrame 
import EmoPy 
from PIL import Image
from Emotion.src.fermodel import FERModel


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

    Emotions = DataFrame(Emotions)
    
    models = {}
    for i in range(len(Emotions)):
        target = list(Emotions.iloc[i, :])
        if i < len(Emotions)-1:
            models['Model' + str(i)] = FERModel(target, verbose=True)

        else:
            models['Model' + str(i)] = FERModel(target[:-1], verbose=True)
    return models





if __name__=='main':
    
    model= Model(['happiness', 'anger'])
    
