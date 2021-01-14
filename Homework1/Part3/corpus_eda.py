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

# %%###############################################################
### Function to display top 30 keys of dictionary based on value
###################################################################

def dict_sorter(dct, num_display):
    ## Make sure given obj is dict
    dct = dict(dct)
    ## sort object into list - high to low
    sorted_dict_list = sorted(
        dct.items(), key=lambda x: x[1], reverse=True)
    ## print out key,value for top num_display based on the value
    for item in sorted_dict_list[0:num_display]:
	    print(item[0], item[1])



# %%###############################################################
## Function to calc bigram PMI based on threshold of bigram count
###################################################################

def pmi_bigram_thresh(bigram_counts, unigram_counts, threshold):
    ## Keep bigram if greater than threshold
    culled_bi_counts = {key: value for key,
                        value in bigram_counts.items() if value > threshold}
    ## Total num bigrams ... so use later to calc bigram probability
    total_bigram = len(culled_bi_counts)
    ## Create new dict for biram prob and calculate based on bigram total
    bigrams_prob = culled_bi_counts
    for key in bigrams_prob:
        bigrams_prob[key] /= total_bigram

    ## Calculate PMI based on bigram prob and unigram prob
    pmi = {}
    bigrams_pairs = list(bigrams_prob.keys())
    for pair in bigrams_pairs:
        w1w2 = bigrams_prob[pair]
        w1 = unigram_counts[pair[0]]
        w2 = unigram_counts[pair[1]]
        pmi_w1w2 = w1w2 / (w1*w2)
        pmi[pair] = pmi_w1w2
    ## Print out result for top 30 results
    dict_sorter(pmi, 30)
    #return pmi



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
## EDA in corpus - Word counting & distribution
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

## Print top 30 words
dict_sorter(type_counts, 30)


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
print(len(stopwords))
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



#%%###############################################################################################
## EDA in corpus - Word association metrics
##############################################################################################

## Calculate unigram probabilities by divide by # of unigrams in corpus
corpus_unigrams_prob = sorted_type_counts_dict
for key in corpus_unigrams_prob:
    corpus_unigrams_prob[key] /=66481

## Sort and print out Unigram probabilities
dict_sorter(corpus_unigrams_prob, 30)

## Get all the bigrams for the corpus
corpus_bigrams = nltk.bigrams(total_corpus)
corpus_bigrams = list(corpus_bigrams)

print("Total bigrams")
total_bigrams = len(corpus_bigrams)

print(total_bigrams)

## Get counts of all bigrams
counts_bigrams = dict(Counter(corpus_bigrams))

dict_sorter(counts_bigrams, 30)

## Calculate bigram probabilities by divide by total num of bigrams
corpus_bigrams_prob = counts_bigrams
for key in corpus_bigrams_prob:
    corpus_bigrams_prob[key] /= 15710622

dict_sorter(corpus_bigrams_prob, 30)

# %%
## 1) examine the 30 highest-PMI word pairs, along with their unigram and bigram frequencies. What do you notice?

pmi_corpus = {}

## Get all bigram pairs and calc PMI
bigrams_pairs = list(corpus_bigrams_prob.keys())
for pair in bigrams_pairs:
    w1w2 = corpus_bigrams_prob[pair]
    w1 = corpus_unigrams_prob[pair[0]]
    w2 = corpus_unigrams_prob[pair[1]]
    pmi_w1w2 = w1w2 / (w1*w2)
    pmi_corpus[pair] = pmi_w1w2
## print out top 30 PMI    
dict_sorter(pmi_corpus, 30)


# %% 2) Experiment with a few different threshold values, and report on what you observe.
### HW Q3: With a threshold of 100, what are the 10 highest-PMI word pairs?
### HW Q4: With a threshold of 100, what are the 10 highest-PMI word pairs?
counts_bigrams = dict(Counter(corpus_bigrams))
print("threshold 10")
pmi_bigram_thresh(counts_bigrams, corpus_unigrams_prob, 10)

print("\n threshold 100")
pmi_bigram_thresh(counts_bigrams, corpus_unigrams_prob, 100)

print("\n threshold 1000")
pmi_bigram_thresh(counts_bigrams, corpus_unigrams_prob, 1000)

print("\n threshold 10,000")
pmi_bigram_thresh(counts_bigrams, corpus_unigrams_prob, 10000)

print("\n threshold 100,000")
pmi_bigram_thresh(counts_bigrams, corpus_unigrams_prob, 100000)

