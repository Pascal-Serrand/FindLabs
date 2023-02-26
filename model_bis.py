from fastapi import FastAPI
import numpy as np 
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from ftraitext import transform_bow_lem_fct




def predict(input_txt):
    vectorizer = joblib.load('tfidfvectorizer.joblib')
    model = joblib.load('model.joblib')
    mlb = joblib.load('mlb.joblib')
#   input_txt = "Earlier using io 6.1 project recently switched / io for lot + change knew updated code but observed strange behavior view every screen get hidden navigation bar repositioning view solves problem ios7 creates problem older io versions. can anyone explain reason happen what changed io causing problem any help would appreciated"
    texte_bow = transform_bow_lem_fct(input_txt)
    texte_vec = vectorizer.transform([texte_bow])
    y_proba = model.predict_proba(texte_vec)
    top_n_labels_idx = np.argsort(-y_proba, axis=-1)[:, :5]
    lst = []
    for i in range(10516):
        lst.append(0)
        if i in top_n_labels_idx:
            lst[i]=1
    res = np.array([lst])
    top_n_labels = mlb.inverse_transform(res)

    return {top_n_labels}
    