from fastapi import FastAPI
import numpy as np 
import joblib
from ftraitext import tokenizer_fct, stop_word_filter_fct, lower_start_fct, lemma_fct

app = FastAPI()

vectorizer = joblib.load('tfidfvectorizer.joblib')
model = joblib.load('model.joblib')
mlb = joblib.load('mlb.joblib')

@app.get("/")
async def predict_labels(input_txt:str,top_labels:int):
    word_tokens = tokenizer_fct(input_txt)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    texte_bow = ' '.join(lem_w)
    texte_vec = vectorizer.transform([texte_bow])
    y_proba = model.predict_proba(texte_vec)
    top_n_labels_idx = np.argsort(y_proba, axis=1)[:,-top_labels :]  
    lst = []
    for i in range(y_proba.shape[1]):
        lst.append(0)
        if i in top_n_labels_idx:
            lst[i]=1
    res = np.array([lst])
    top_n_labels = mlb.inverse_transform(res)   
    return {'prediction labels' : top_n_labels}
