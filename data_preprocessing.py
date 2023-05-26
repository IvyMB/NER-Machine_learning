import numpy as np
import pandas as pd
import string
import re
import random

with open('businessCard.txt', mode='r', encoding='utf8', errors='ignore') as f:
    text = f.read()

data = list(map(lambda x: x.split('\t'), text.split('\n')))
df = pd.DataFrame(data[1:], columns=data[0])

df.head(10)

whitespace = string.whitespace
punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
tableWhitespace = str.maketrans('', '', whitespace)
tablePunctuation = str.maketrans('', '', punctuation)


def cleanText(txt):
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)

    return str(removepunctuation)

df['text'] = df['text'].apply(cleanText)

df = df.query("text != '' ")
df.dropna(inplace=True)

df.head(10)

group = df.groupby(by='id')

cards = group.groups.keys()
allCardsData = []
for card in cards:
    cardData = []
    grouparray = group.get_group(card)[['text', 'tag']].values
    content = ''
    annotations = {'entities': []}
    start = 0
    end = 0
    for text, label in grouparray:
        text = str(text)
        stringLength = len(text) + 1

        start = end
        end = start + stringLength

        if label != 'O':
            annot = (start, end - 1, label)
            annotations['entities'].append(annot)

        content = content + text + ' '

    cardData = (content, annotations)
    allCardsData.append(cardData)

allCardsData

random.shuffle(allCardsData)


len(allCardsData)

TrainData = allCardsData[:240]
TestData = allCardsData[240:]

import pickle

pickle.dump(TrainData,open('./data/TrainData.pickle', mode='wb'))
pickle.dump(TestData,open('./data/TestData.pickle', mode='wb'))




