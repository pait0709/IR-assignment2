import pickle
tf_idf = {}
with open('tf_idf.pkl', 'rb') as f:
    tf_idf=pickle.load(f)

DocumentFrequency = {}
with open('DocumentFreq.pkl', 'rb') as f:
    DocumentFrequency=    pickle.load(f)

total_vocab=[x for x in DocumentFrequency]

def doc_freq(word):
    c = 0
    try:
        c = DocumentFrequency[word]
    except:
        pass
    return c

import pandas as pd

df=pd.read_csv('A2_Data.csv')
N=len(df)
total_vocab_size=len(total_vocab)

import numpy as np
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    ind = total_vocab.index(i[1])
    D[i[0]][ind] = tf_idf[i]


import math
import os
from PIL import Image
from io import BytesIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from collections import Counter
from scipy.spatial.distance import cosine
import cv2
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
def gen_vector(tokens):

    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

def preprocess_text(text):

    text = text.lower()

    tokens = word_tokenize(text)


    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]


    tokens = [token for token in tokens if token not in string.punctuation]

    tokens = [token for token in tokens if token.isalnum()]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    return lemmatized_tokens

def cosine_similarity(k, query):

    tokens = preprocess_text(query)

    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    return out

file = open("query.txt", "r")
query = file.read()
file.close()

query=query.split('\n')
os.chdir('/Users/pait/Desktop/Coding/IR')
#image retrieval
index_of_features = {}
with open('index_of_features.pkl', 'rb') as f:
    index_of_features=pickle.load(f)

import requests
url=query[0]
response = requests.get(url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    #save in images folder

    

    img.save(f'query.jpg')
else:
    print("Invalid URL")

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    #show image
    return image

def extract_features(image):
    model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    image = preprocess_input(image)
    features = model.predict(np.expand_dims(image, axis=0))
    return features.flatten()

def normalize_features(features):
    normalized_features = (features - np.mean(features)) / np.std(features)
    return normalized_features

def retrieve_similar_images(query_features, database_features, top_k=5):
    similarity_scores=[]
    for img_path, features in database_features.items():

        similarity = 1-cosine(query_features, features)
        similarity_scores.append((img_path, similarity))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return similarity_scores[:top_k]

query_image_path = "query.jpg"
query_image = preprocess_image(query_image_path)
query_features = extract_features(query_image)
query_features_normalized = normalize_features(query_features)

top_indices = retrieve_similar_images(query_features_normalized, index_of_features)



##image retrieval starts here
image_number_to_product_id={}
with open ('image_number_to_product_id.pkl', 'rb') as f:
    image_number_to_product_id=pickle.load(f)

result={}

for i in top_indices:
    image_num=i[0].split('.')[0]
    image_num=int(image_num)
    try:
        result[image_number_to_product_id[image_num]]=max(result[image_number_to_product_id[image_num]],i[1])
    except:
        result[image_number_to_product_id[image_num]]=i[1]
    if len(result)==3:
        break

composite_retrieval={}

with open ('result.txt', 'w') as f:
    f.write("Image Retrieval\n")
    for i in result.keys():
        for j in range (0,N):
            if df['Product_id'][j]==i:
                ImageURL=df['Image'][j]
                Review_text=df['Review Text'][j]
                f.write("Product ID: ")
                f.write(str(i))
                f.write('\n')
                f.write("Image URL: ")
                f.write(ImageURL)
                f.write('\n')
                f.write("Review Text: ")
                f.write(Review_text)
                f.write('\n')
                f.write("Cosine Similarity of image: ")
                f.write(str(result[i]))
                f.write('\n')
                f.write("Cosine Similarity of text ")
                token_query=preprocess_text(query[1])
                token_review=preprocess_text(Review_text)
                f.write(str(cosine_sim(gen_vector(token_query),gen_vector(token_review))))
                f.write('\n')
                f.write("composite score: ")
                f.write(str((result[i]+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2))
                try:
                    composite_retrieval[i]=max(composite_retrieval[i],(result[i]+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2)
                except:
                    composite_retrieval[i]=(result[i]+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2
                f.write('\n')
                break

f.close()
#text retrieval

def process_URLS(ImageURLs):
    query_image_path = "query.jpg"
    query_image = preprocess_image(query_image_path)
    query_features = extract_features(query_image)
    query_features_normalized = normalize_features(query_features)
    result=0
    for i in range(len(ImageURLs)):
        response = requests.get(ImageURLs[i])
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(f'images/sample{i}.jpg')
            image = preprocess_image(f'images/sample{i}.jpg')
            features = extract_features(image)
            features_normalized = normalize_features(features)
            result=max(cosine_sim(query_features_normalized,features_normalized),result)

    return result


k=3
query = query[1]
import ast
out = cosine_similarity(k, query)
with open ('result.txt', 'a') as f:
    f.write("\nText Retrieval\n")
    for i in out:
        ImageURL=df['Image'][i]
        Review_text=df['Review Text'][i]
        f.write("Product ID: ")
        f.write(str(df['Product_id'][i]))
        f.write('\n')
        f.write("Image URL: ")
        f.write(ImageURL)
        f.write('\n')
        f.write("Review Text: ")
        f.write(Review_text)
        f.write('\n')
        f.write("Cosine Similarity of text: ")
        token_query=preprocess_text(query)
        token_review=preprocess_text(Review_text)
        f.write(str(cosine_sim(gen_vector(token_query),gen_vector(token_review))))
        f.write('\n')
        url_list = ast.literal_eval(ImageURL)
        result=process_URLS(url_list)
        f.write("Cosine Similarity of Image ")
        f.write(str(result))
        f.write('\n')
        f.write("composite score: ")
        f.write(str((result+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2))
        try:
            composite_retrieval[df['Product_id'][i]]=max(composite_retrieval[df['Product_id'][i]],(result+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2)
        except:
            composite_retrieval[df['Product_id'][i]]=(result+cosine_sim(gen_vector(token_query),gen_vector(token_review)))/2
        f.write('\n')

def highest_3_values_1(my_dict):
    result_dict = {}
    sorted_dict = sorted(my_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(3):
        result_dict[sorted_dict[i][0]] = sorted_dict[i][1]
    return result_dict
 
composite_retrieval=highest_3_values_1(composite_retrieval)

with open ('result.txt', 'a') as f:
    g=0
    f.write("\nCombined Retrieval\n")
    for i in composite_retrieval.keys():
        if g==3:
            break
        for j in range (0,N):
            if df['Product_id'][j]==i:
                ImageURL=df['Image'][j]
                Review_text=df['Review Text'][j]
                f.write("Product ID: ")
                f.write(str(i))
                f.write('\n')
                f.write("Image URL: ")
                f.write(ImageURL)
                f.write('\n')
                f.write("Review Text: ")
                f.write(Review_text)
                f.write('\n')
                f.write("composite score: ")
                f.write(str(composite_retrieval[i]))
                f.write('\n')
                break
        g+=1
    

f.close()
print("Done")
