#!/usr/bin/env python3


#%%###########################################################################################
## Package imports
##############################################################################################

import nltk
#from lxml import etree
#import gzip
#import sys
#import re
#import string
from collections import Counter
import matplotlib.pyplot as plt
from scipy import special
import numpy as np

path = "/Users/nguyjust/CS662_NLP/Homework1/"



#%%###########################################################################################
## Read in data -- Word counting & distribution
##############################################################################################

## Read in the data that was restructured based on sentence
with open(path + 'corpus_word_tok.txt', 'r') as fh:
    data = fh.readlines()
    corp_sentence_tok = [line.strip() for line in data]

print("read in data")
print(len(corp_sentence_tok))

corp_word_tok = [sentence.split(" ") for sentence in corp_sentence_tok]

## Print checks for correct read in/separation of data
print(len(corp_sentence_tok))
print(corp_sentence_tok[0:2])
print(len(corp_word_tok))
print(corp_word_tok[0:2])



#%%###############################################################################################
## EDA in corpus
##############################################################################################

# Put all words into a single list
total_corpus = [word for sentence in corp_word_tok for word in sentence]

## Only unique words
total_types = set(total_corpus)

## 1) How many unique *types* present in corpus?
print("Unique types in corpus")
print(len(total_types))


## 2) How many *unigram tokens*?
print("Unigram tokens")
print(len(total_corpus))


#%%# 4) What are the thirty most common words?
## I tried to iterate through the unique words and count them from the corpus, but this was too slow
#word_counts = [(word, total_corpus.count(word)) for word in total_types]

## Count all words in corpus, putting them into dict
type_counts = dict(Counter(total_corpus))

## Check unique types after counting and the data type!
#print(len(type_counts))
#print(type(type_counts))
#type_counts.sort(reverse=True)

## Sort the dictionary based on the values (of counts)
sorted_type_counts = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
## Print out top 30 most occurring words
for word in sorted_type_counts[0:30]:
	print(word[0], word[1])


#%% 3) Produce a rank-frequency plot (similar to those seen on the Wikipedia page for Zipf’s Law) for this corpus.

## Put sorted counts/words back into dict!
sorted_type_counts_dict = dict(sorted_type_counts)
#convert value of frequency to numpy array
freq_counts = np.array(list(sorted_type_counts_dict.values()))

freq_counts = np.array(freq_counts)
ranks = np.array(range(1, len(freq_counts) + 1))

plt.loglog(ranks, freq_counts)
plt.xlabel("Log(Rank)")
plt.ylabel("Log(Frequency)")
plt.show()


#%% 5) What happens to your type/token counts if you remove stopwords using nltk.corpora’s stopwords list?

#nltk.download('stopwords')
## Get the nltk stopwords into a set
stopwords = list(nltk.corpus.stopwords.words('english'))
#print(stopwords)
print("NLTK number of stopwords")
print(len(stopwords))ß
## Convert stop words to upper to match my corpus
stopwords = [word.upper() for word in stopwords]

## New dict containing all counts of corpus... So i can delete the stopwords
type_counts_noStop = type_counts

## Iterate through stopwords and remove it from dict if it exists
for word in stopwords:
    if word in type_counts_noStop:
        type_counts_noStop.pop(word)

print("Type count")
print(len(type_counts_noStop))

print("Token count")
print(sum(type_counts_noStop.values()))


# 6) After removing stopwords, what are the thirty most common words?

## Sort the dictionary based on the values (of counts)
sorted_type_counts_noStop = sorted(type_counts_noStop.items(), key=lambda x: x[1], reverse=True)
## Print out top 30 most occurring words
for word in sorted_type_counts_noStop[0:30]:
	print(word[0], word[1])

# %%
print(stopwords)



