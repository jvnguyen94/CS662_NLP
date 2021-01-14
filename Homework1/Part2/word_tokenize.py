#!/usr/bin/env python3


#%%###########################################################################################
## Package imports
##############################################################################################

import nltk
#from lxml import etree
#import gzip
import sys
import re
import string

path = "/Users/nguyjust/CS662_NLP/Homework1/"



#%%###########################################################################################
## Read in data
##############################################################################################

## Read in the data that was restructured based on sentence
fh = open(path + "deserialized.txt", 'r')
data = fh.read()
fh.close()

print("read in data")

#%%###############################################################################################
## Tokenize by sentence and word
##############################################################################################

## Look at the data type
#print(type(data))

## Run nltk.sent_tokenize to split into sentences
data_sentence_tok = nltk.sent_tokenize(data)
print("Number of sentences in corpus:" + str(len(data_sentence_tok)))

## Run nltk.word_tokenize to split into words
data_word_tok = [nltk.word_tokenize(sentence) for sentence in data_sentence_tok]
print('finish word tokenize')

## Double check that the number of sentences is still intact
print(len(data_word_tok))
#%%

## Write out tokenized words/sentences
fh = open('corpus_word_tok.txt', 'a')
## Loop through each sentence
for sentence in data_word_tok:
	## Caps all words if the word is ALPHA to remove punctuation
	sorted_sentence = [item.upper() for item in sentence if item.isalpha()]
	## Write out
	for ii in sorted_sentence:
		fh.write(str(ii) + " ")
	fh.write('\n')
## Close file
fh.close()


