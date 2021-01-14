#!/usr/bin/env python3


##############################################################################################
## Package imports
##############################################################################################

import nltk
from lxml import etree
import gzip
import sys



##############################################################################################
## Import system vars for filenames
##############################################################################################

## Filename from system
xml_filename = sys.argv[1]

## Check file name from sys arg
#print(xml_filename)



##############################################################################################
## Deserialization script
##############################################################################################

## Read in file
fh = gzip.open(xml_filename, 'rb')

file_content = fh.read()

## Check for read initial read in
#print(file_content)
fh.close()

## Put data into XML tree so it can be parsed
root = etree.XML(file_content)

## Check data in XML parse
#print(root)
#print(etree.tostring(root, pretty_print=True))

## Use XPath for parsing for paragraph text -- if the DOC == 'story'
paragraph = root.xpath("//DOC[@type='story']/TEXT/P/text()")

## Check for grabbing the tags in correct parts of tree
#print(paragraph)
#print(len(paragraph))
#print(paragraph[0])
#print(paragraph[-1])



##############################################################################################
## Data write out
##############################################################################################

## Loop through list and write out data to file, appending
with open('deserialized.txt', 'a') as fh:
    for item in paragraph:
        fh.write("%s\n" % item)
fh.close()



