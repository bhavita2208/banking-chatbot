import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemtzr = WordNetLemmatizer()
contents = json.loads(open('data.json').read())

words = pickle.load(open('word.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbotmodel.model')

def clean_up(line):
    line_words = nltk.word_tokenize(line)
    line_words = [lemtzr.lemmatize(word) for word in line_words]
    return line_words

def group_words(line):
    line_words = clean_up(line)
    group = [0] * len(words)
    for w  in line_words:
        for i, word in enumerate(words):
            if word == w :
                group[i] = 1
    return np.array(group)

def pred_class(line):
    gw =group_words(line)
    res = model.predict(np.array([gw]))[0]
    Er_threshold = 0.25
    results = [[i,r] for i,r in enumerate(res) if r> Er_threshold]
    results.sort(key= lambda x:x[1], reverse=True)
    return_lst = []
    for r in results:
        return_lst.append({'content':classes[r[0]], 'probability ': str(r[1])})
    return return_lst

def get_resp(content_lst, content_json):
    tag = content_lst[0]['content']
    list_contents = content_json['content']
    for i in list_contents:
        if i['tag'] == tag:
            result = random.choice(i['answers'])
            break
    return result

while True:
    message = input("")
    ints = pred_class(message)
    res = get_resp(ints, contents)
    print(res)


