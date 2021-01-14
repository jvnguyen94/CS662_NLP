#!/usr/bin/env python3

##############################################################################################
## Package imports
##############################################################################################

import nltk
#from lxml import etree
#import gzip
import sys
import re

path = "/Users/nguyjust/CS662_NLP/Homework1/"



##############################################################################################
## Read in data 
##############################################################################################

## Read in the data that was deserialized
fh = open(path + "deserialized.txt", 'r')
data = fh.read()
fh.close()



##############################################################################################
## Sentence clean up
##############################################################################################

## Remove all the random newline characters 
no_newline = data.replace("\n", " ")

## Cannot split on periods... does not handle numbers (ie 1.4) or abbrev
#sentence_split = no_newline.split('.')

## Transform to all caps
all_caps = no_newline.upper()

## Attempt at split sentence based on regex
## below is still too naive... problem does not handle abbreviations but can handle decimal numbers
#sentence_regex = r"\.(\s)"

## Regex to look before the period to make sure it is not abbreviation and split on either ./?/!
sentence_regex = r"(?<!\..)[.?!]\s+"
sentence_split = re.split(sentence_regex, all_caps)



##############################################################################################
## Write out cleaned sentences
##############################################################################################

with open('restruc_deserialized.txt', 'w') as fh:
	for item in sentence_split:
		## Remove write out if it is just a blank
		if item != " ":
			fh.write("%s\n" % item)
fh.close



