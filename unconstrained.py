import csv
import spacy 
from tabulate import tabulate
from scipy.stats import spearmanr
import pandas as pd


corpus = spacy.load('en_core_web_lg')
simi = []



with open('sample.txt', 'r') as file:

    txt = csv.reader(file, delimiter='\t')
    next(txt)
    rows=[]
    
    for row in txt:
        word1 = row[0]
        word2 = row[1]
        tokens = corpus(word1 + " " + word2)
        sim = tokens[0].similarity(tokens[1])
        rows.append([word1, word2, sim])
        simi.append(sim)

    print(tabulate(rows, headers=["Word 1", "Word 2", "Similarity"]))

simlex_subset = pd.read_csv('sample.txt', sep='\t')
simlex_subset = pd.read_csv('sample.txt', sep='\t')
ground_truth_scores = simlex_subset['SimLex999'].values
# print(ground_truth_scores)

correlation, _ = spearmanr(simi, ground_truth_scores)

print(f'Spearman Correlation: {correlation*100}%')

