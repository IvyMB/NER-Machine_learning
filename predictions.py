import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
from spacy import displacy


def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)

    return str(removepunctuation)

warnings.filterwarnings('ignore')

### Load NER model
model_ner = spacy.load('./output/model-best/')

# Load Image
image = cv2.imread('./data/6.jpg')

# cv2.imshow('businesscard',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# extract data using Pytesseract
tessData = pytesseract.image_to_data(image)
# convert into dataframe
tessList = list(map(lambda x: x.split('\t'), tessData.split('\n')))
df = pd.DataFrame(tessList[1:], columns=tessList[0])
df.dropna(inplace=True)  # drop missing values

df['text'] = df['text'].apply(cleanText)

# convet data into content
df_clean = df.query('text != "" ')
content = " ".join([w for w in df_clean['text']])
# get prediction from NER model
doc = model_ner(content)

# Renderizar os resultados da predição
#html = displacy.render(doc, style='ent')
#with open('output.html', 'w', encoding='utf-8') as file:
 #   file.write(html)


docjson = doc.to_json()
docjson.keys()
doc_text = docjson['text']

datafram_tokens = pd.DataFrame(docjson['tokens'])
# Spliting doc_text com os valores de 'start' e 'end' e criando uma nova coluna chamada text
datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(
    lambda x: doc_text[x[0]:x[1]], axis=1)
datafram_tokens.head(10)


right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]

# Combinar os dois dataframes com left join
datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')


# Preenchar os NaN valores por O
datafram_tokens.fillna('O', inplace=True)
print(datafram_tokens)

