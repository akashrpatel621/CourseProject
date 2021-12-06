import requests
import json
import pandas as pd
from csv import writer
import datetime
import nltk
from nltk.corpus import movie_reviews
import numpy as np # linear algebra
import os
import warnings
warnings.simplefilter('ignore')
import seaborn as sns
sns.set()
import random
from pprint import pprint
from datetime import datetime
import collections
import re
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from requests.models import Response
def main(stock):
    if(stock == ""):
        return

    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAGPOVwEAAAAA4TbpZHwB6%2FVrgW8WoHvHFvMmZJU%3DHshKMN00NxazKkbH2TGss2NZNCaCcCXLJsQL2B8AwwnIqPfEWQ"
    #define search twitter function
    def search_twitter(query, tweet_fields, bearer_token = BEARER_TOKEN):
        headers = {"Authorization": "Bearer {}".format(bearer_token)}
        x = "https://api.twitter.com/2/tweets/search/recent?query={}&{}&max_results=100&&start_time=" + str(datetime.now().date()) + "T00:00:00Z"
        url = x.format(
            query, tweet_fields
        )
        response = requests.request("GET", url, headers=headers)

        print(response.status_code)

        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()

    #search term
    query = stock
   
    #twitter fields to be returned by api call
    tweet_fields = "tweet.fields=text,author_id,created_at"

    #twitter api call
    json_response = search_twitter(query=query, tweet_fields=tweet_fields, bearer_token=BEARER_TOKEN)
    #pretty printing

    data = json_response['data']
    import csv
   
    t = []
    # close the file
    for i in range(len(data)):
        t.append(data[i]['text'])
  
    # switch filename to local path where dataset is saved
    file_name = "C:\\Users\\akashpatel\\Downloads\\tweets_labelled_09042020_16072020.csv"
    data = pd.read_csv(file_name, sep=';').set_index('id')

    # Preprocess for runnning models
    # delete emoji, handle, url
    # delete stop_word including RT
    ticker_pattern = re.compile(r'(^\$[A-Z]+|^\$ES_F)')
    ht_pattern = re.compile(r'#\w+')

    ticker_dic = collections.defaultdict(int)
    ht_dic = collections.defaultdict(int)

    for text in data['text']:
        for word in text.split():
            if ticker_pattern.fullmatch(word) is not None:
                ticker_dic[word[1:]] += 1
            
            word = word.lower()
            if ht_pattern.fullmatch(word) is not None:
                ht_dic[word] += 1

    charonly = re.compile(r'[^a-zA-Z\s]')
    handle_pattern = re.compile(r'@\w+')
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    url_pattern = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    pic_pattern = re.compile('pic\.twitter\.com/.{10}')
    special_code = re.compile(r'(&amp;|&gt;|&lt;)')
    tag_pattern = re.compile(r'<.*?>')

    STOPWORDS = set(stopwords.words('english')).union(
        {'rt', 'retweet', 'RT', 'Retweet', 'RETWEET'})

    lemmatizer = WordNetLemmatizer()

    def hashtag(phrase):
        return ht_pattern.sub(' ', phrase)

    def remove_ticker(phrase):
        return ticker_pattern.sub('', phrase)
        
    def specialcode(phrase):
        return special_code.sub(' ', phrase)

    def emoji(phrase):
        return emoji_pattern.sub(' ', phrase)

    def url(phrase):
        return url_pattern.sub('', phrase)

    def pic(phrase):
        return pic_pattern.sub('', phrase)

    def html_tag(phrase):
        return tag_pattern.sub(' ', phrase)

    def handle(phrase):
        return handle_pattern.sub('', phrase)

    def decontracted(phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        
        # DIS, ticker symbol of Disney, is interpreted as the plural of "DI" 
        # in WordCloud, so I converted it to Disney
        phrase = re.sub('DIS', 'Disney', phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"(he|He)\'s", "he is", phrase)
        phrase = re.sub(r"(she|She)\'s", "she is", phrase)
        phrase = re.sub(r"(it|It)\'s", "it is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"(\'ve|has)", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def onlychar(phrase):
        return charonly.sub('', phrase)

    def remove_stopwords(phrase):
        return " ".join([word for word in str(phrase).split()\
                        if word not in STOPWORDS])

    def tokenize_stem(phrase):   
        tokens = word_tokenize(phrase)
        stem_words =[]
        for token in tokens:
            word = lemmatizer.lemmatize(token)
            stem_words.append(word)        
        buf = ' '.join(stem_words)    
        return buf

    def arrange_text(ds):
        ds['text2'] = ds['text'].apply(emoji)
        ds['text2'] = ds['text2'].apply(handle)
        ds['text2'] = ds['text2'].apply(specialcode)
        ds['text2'] = ds['text2'].apply(hashtag)
        ds['text2'] = ds['text2'].apply(url)
        ds['text2'] = ds['text2'].apply(pic)
        ds['text2'] = ds['text2'].apply(html_tag)
        ds['text2'] = ds['text2'].apply(onlychar)
        ds['text2'] = ds['text2'].apply(decontracted)
        ds['text2'] = ds['text2'].apply(onlychar)
        ds['text2'] = ds['text2'].apply(tokenize_stem)
        ds['text2'] = ds['text2'].apply(remove_stopwords)

    arrange_text(data)
    train = data[data['sentiment'] == data['sentiment']]

    # Formats the tweet as a tuple of token of words and its sentiment 
    def format_df(df):
        docs = []
        for row in df.iterrows():
            tmp = row[1]['text'].split()
            docs.append((tmp, row[1]['sentiment']))
        return docs
    docs = format_df(train)

    random.shuffle(docs)

    # Creates a dictionary with word counts in the document 
    def word_counts(docs):
        words = Counter()
        for pair in docs:
            words = sum((words, Counter(pair[0])), Counter())
        return words

    words = word_counts(docs)

    word_features = list(words)[:2000]

    # Formats the data for nltk's testing and training purposes
    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    # Define the feature extractor

    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000]


    documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # combine the movie reviews and twitter dataset
    dd = docs + documents
    check = [t[1] for t in dd]
    print(set(check))
    random.shuffle(dd)

    featuresets = [(document_features(d), c) for (d,c) in dd]

    classifier = nltk.NaiveBayesClassifier.train(featuresets)


    test_set = pd.DataFrame({"text":t})
    def testformat_df(df):
        docs = []
        for row in df.iterrows():
            tmp = row[1]['text'].split()
            docs.append(tmp)
        return docs

    test = testformat_df(test_set)
    random.shuffle(test)
    testfeaturesets = [document_features(d) for (d) in test]
    a = []
    for t in testfeaturesets:
        a.append(classifier.classify(t))

    # aggregate the results as percentages for sentiments 

    a = Counter(a)
    return a