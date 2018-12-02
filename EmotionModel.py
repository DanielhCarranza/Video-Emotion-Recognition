import numpy as np
from pandas import DataFrame 
import EmoPy 
from PIL import Image
from EmoPy.src.fermodel import FERModel
import videoCapture 

def Model(target_emotions):
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


    model = FERModel(target_emotions, verbose=True)

    return model
def load_image(Path):
    return Image.open(Path)

def predict(model, image):
    print('Predicting  image...')
    return model.predict(image)


if __name__=='main':
    
    model= Model(['happiness', 'anger'])
    img= load_image('happy.jpg')
    predict(model, img)