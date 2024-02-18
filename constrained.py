from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import csv
from tabulate import tabulate
from scipy.stats import spearmanr
import numpy as np

simi = []

with open('Pride.txt', 'r') as file:
    text = file.read()

tokens=word_tokenize(text)
# tokens = word_tokenize(text)
# print(len(tokens))
# tokens.extend(tokens1)
# print(len(tokens),len(tokens1))
model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)


def w2v(text,model):
    
    # words = process(text)
    words = text
    w_vector = []
    # for word in words:
    #     if word in model.key_to_index:
    #         w_vector.append(model[word])
    if words in  model.wv:
        w_vector.append(model.wv[words])
    if len(w_vector) == 0:
        return np.zeros(model.vector_size)
    else:
        sen_vec = np.mean(w_vector,axis=0)
        return sen_vec
    
def similarity(vec1,vec2):
    
    if np.linalg.norm(vec1) >0 and np.linalg.norm(vec2)>0:
    
        return (np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    else:
        return 0
    

# def wordnet_similarity(word1, word2):
    
#     synsets1 = wn.synsets(word1)
#     synsets2 = wn.synsets(word2)

#     sim_scores=[]

#     if not synsets1 or not synsets2:
#         return 0

#     for syn1 in synsets1:
#         for syn2 in synsets2:
#             if syn1.path_similarity(syn2) is not None:
#                 sim_scores.append(syn1.path_similarity(syn2))

#     if sim_scores:
#         return max(sim_scores)
#     else:
#         return 0




with open('sample.txt', 'r') as file:

    txt = csv.reader(file, delimiter='\t')
    next(txt)
    rows=[]
    
    for row in txt:
        word1 = row[0]
        word2 = row[1]

        # sim = wordnet_similarity(word1,word2)
        vec1 = w2v(word1,model)
        vec2 = w2v(word2,model)
        sim = similarity(vec1,vec2)
        rows.append([word1, word2, sim])
        simi.append(sim)

    print(tabulate(rows, headers=["Word 1", "Word 2", "Similarity"]))



simlex_subset = pd.read_csv('sample.txt', sep='\t')



simlex_subset = pd.read_csv('sample.txt', sep='\t')
# print(simlex_subset)


ground_truth_scores = simlex_subset['SimLex999'].values

correlation, _ = spearmanr(simi, ground_truth_scores)

print(f'Spearman Correlation: {correlation*100}%')