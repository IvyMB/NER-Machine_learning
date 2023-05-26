import numpy as np
import pandas as pd
import cv2
import pytesseract

import os
from glob import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


img_paths = glob('./Selected/*.jpeg')

all_business_card = pd.DataFrame(columns=['id', 'text'])

for img_path in tqdm(img_paths, desc='BusinessCard'):
    # imgPath = imgPaths[0]
    _, filename = os.path.split(img_path)
    # extract data and text
    image = cv2.imread(img_path)
    data = pytesseract.image_to_data(image)
    data_list = list(map(lambda x: x.split('\t'), data.split('\n')))
    df = pd.DataFrame(data_list[1:], columns=data_list[0])
    df.dropna(inplace=True)
    df['conf'] = df['conf'].astype(int)

    useful_data = df.query('conf >= 30')

    # Dataframe
    business_card = pd.DataFrame()
    business_card['text'] = useful_data['text']
    business_card['id'] = filename

    # concatenation
    all_business_card = pd.concat((all_business_card, business_card))

all_business_card.to_csv('businessCard.csv', index=False)
