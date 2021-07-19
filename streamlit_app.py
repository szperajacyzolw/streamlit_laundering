from boruta import BorutaPy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from joblib import load
import nltk
import enchant  # spellchecker
import requests
import streamlit as st
import os
import re
import numpy as np
import pandas as pd


with st.spinner('Downloading dependencies...'):
    nltk.download('stopwords')
    nltk.download('point')
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import NOUN, ADJ, VERB, ADV
from nltk.stem.wordnet import WordNetLemmatizer


this_dir = os.getcwd()
save_dir = os.path.join(this_dir, 'saves')

lemmatizer = WordNetLemmatizer()
english_spellcheck = enchant.Dict('en_US')
stopwords = stopwords.words('english')


def get_lemma(word: str) -> str:
    'returns lemmatized word based on nltk.'
    tag = pos_tag([word])[0][1][0].upper()
    tags = {'J': ADJ, 'N': NOUN, 'V': VERB, 'R': ADV}

    return lemmatizer.lemmatize(word, tags.get(tag, NOUN))


def text_cleaner(text: list) -> str:
    '''
    takes list of strings(an article), produces string
    without short expressions, numbers, stop words and punctuation,
    all in lowercase
    '''

    text = text.copy()

    # remove punctuation, shorts, broken words and digits
    # short words are often without particular meaning or are residuals of longer, missing words
    punct_dig = re.compile(r'[^a-zA-Z\s]')
    shorts = re.compile(r'\b\w{1,3}\b')
    # broken words with numbers and symbols inclusions, eg. bal00n, m0ney, ca$h
    broken = re.compile(r'\b[a-zA-z]+[^\sa-zA-Z]+[a-zA-Z]+\b')
    text_ = []
    for i in range(len(text)):
        br = re.sub(broken, '', text[i])
        pd = re.sub(punct_dig, ' ', br)
        text_.append(re.sub(shorts, '', pd))
    text = text_

    # split elements and lemmatize
    text = [get_lemma(word) for t in text for word in word_tokenize(t)]

    # aggressive lowercase
    text = list(map(str.casefold, text))

    # remove stop words
    text = [elem for elem in text if not elem in stopwords]

    # remove foreign words
    text = [elem for elem in text if english_spellcheck.check(elem)]

    return ' '.join(text)


def text_scraper(http: str) -> (list, list):

    clean_articles = []

    web_text = requests.get(http, timeout=10).text
    soup = BeautifulSoup(web_text, 'html.parser').stripped_strings
    raw = list(soup)
    cleaned = text_cleaner(raw)
    clean_articles.append(cleaned)

    return clean_articles, raw


def _toarray(x: 'sparse array') -> np.array:
    '''
    helper function for pipeline array transformation
    '''
    return x.toarray()


rfc_boruta = RandomForestClassifier(n_jobs=4, class_weight='balanced', max_depth=5)
to_array = FunctionTransformer(_toarray)
boruta = BorutaPy(rfc_boruta, perc=100, verbose=1)


@st.cache(allow_output_mutation=True)
def load_models():
    model_lg = load(os.path.join(save_dir, 'brenoulli_bayes_lg.gz'))
    model_cs = load(os.path.join(save_dir, 'brenoulli_bayes_cs.gz'))

    return model_lg, model_cs


@st.cache(allow_output_mutation=True)
def load_dataframes():
    df_cs = pd.read_csv(os.path.join(save_dir, 'charges_vs_sentence_features_prob.csv'),
                        index_col=0)
    df_lg = pd.read_csv(os.path.join(save_dir, 'generic_vs_laundering_features_prob.csv'),
                        index_col=0)

    return df_lg, df_cs


def news_classifier(http: str, model_stage1: object, model_stage2: object):
    '''
    print out prediction of given http adress
    http - full http://www... string
    model_stage1&2 - GridSearchCV object of laundering vs generic and charges vs sentence models
    '''

    text, _ = text_scraper(http)
    pred_stage1 = model_stage1.predict(text)[0]

    if pred_stage1:
        result = ['charges', 'sentence']
        pred_stage2 = model_stage2.predict(text)[0]
        return f'Provided website mentions money laundering {result[pred_stage2]}'
    else:
        return 'Provided website does not mention money laundering'


with st.spinner('Loading content...'):
    df_lg, df_cs = load_dataframes()
    model_lg, model_cs = load_models()

st.title('Check-for-laundering-content app')

with st.form(key='input_http'):
    http_input = st.text_input(
        label=r'Enter full web link here. Proper format: http://www.site...')
    submit_button = st.form_submit_button(label='Submit')


if submit_button:
    st.markdown(f'**{news_classifier(http_input, model_lg, model_cs)}**')

st.write("Below you can explore model's fetures probabilities given a class")
st.write('Plots are limited to first 50 features')
st.write('In the upper right corner of tables and charts you can find zoom button')

col1, col2 = st.beta_columns(2)

with col1:
    st.header('Model: generic news vs laundering')
    st.dataframe(df_lg)
    fig, ax = plt.subplots(tight_layout=True, figsize=(20, 20))
    ypos = np.arange(50)
    ax.barh(ypos, df_lg.iloc[:50, 0], align='edge', color='g', height=0.4)
    ax.barh(ypos, df_lg.iloc[:50, 1], align='center', color='r', height=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df_lg.index[:50])
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    red_patch = mpatches.Patch(color='r', label='Laundering')
    green_patch = mpatches.Patch(color='g', label='Generic')
    plt.legend(handles=[red_patch, green_patch])
    st.pyplot(fig)

with col2:
    st.header('Model: charges vs sentence')
    st.dataframe(df_cs)
    fig, ax = plt.subplots(tight_layout=True, figsize=(20, 20))
    ypos = np.arange(50)
    ax.barh(ypos, df_cs.iloc[:50, 0], align='edge', color='g', height=0.4)
    ax.barh(ypos, df_cs.iloc[:50, 1], align='center', color='r', height=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(df_cs.index[:50])
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    red_patch = mpatches.Patch(color='r', label='Sentence')
    green_patch = mpatches.Patch(color='g', label='Charges')
    plt.legend(handles=[red_patch, green_patch])
    st.pyplot(fig)
