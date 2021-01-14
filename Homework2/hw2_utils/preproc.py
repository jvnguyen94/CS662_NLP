from collections import Counter

import pandas as pd
import numpy as np

## deliverable 1.1
def bag_of_words(text):
    """
    Count the number of word occurrences for a given document
    
    :param text: a document, as a single string
    :returns: a Counter representing a single document's word counts
    :rtype: Counter
    """
    ## Split into words based on space character
    word_split = text.split(" ")
    ## Count all the words
    word_counts = Counter(word_split) 
    ## Remove empty "words" or space characters that are captured
    del word_counts[" "]
    del word_counts[""]

    return word_counts



## deliverable 1.2
def aggregate_counts(bags_of_words):
    """
    Aggregate bag-of-words word counts across an Iterable of documents into a single bag-of-words.
    
    :param bags_of_words: an iterable of bags of words, produced from the bag_of_words() function above
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    """
    
    ## Initialize counter obj to keep track of counts
    aggregate = Counter()
    ## Iterate through each dict and aggregate them!
    for item in bags_of_words:
        aggregate.update(item)

    return aggregate



## deliverable 1.3
def compute_oov(bow1, bow2):
    """
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    """
    bow1_only = set(bow1) - set(bow2)

    return bow1_only
    
    

## deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    """
    Prune target_data to only include words that occur at least min_counts times in training_counts
    
    :param training_counts: aggregated Counter for the training data -- type: counter object
    :param target_data: list of Counters containing dev bow's -- type: list of counter objects
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype list of Counters, set
    """
    
    ## Keep word if its count is >= than the min_counts (List of words only)
    thresh_words = [item for item in list(training_counts) if training_counts[item] >= min_counts]

    ## Create list to keep track of culled Counter objs
    pruned_vocab_target = []
    ## Loop through each Counter obj and grab only words that are in `pruned_words`
    for counter in target_data:
        pruned_vocab_target.append(Counter({item: counter[item] for item in counter if item in thresh_words}))

    ## Prune to get the list of words in pruned vocab
    pruned_words_counts_train = set([(item, training_counts[item]) for item in thresh_words])
    
    return pruned_vocab_target, pruned_words_counts_train



## Helper functions

def read_data(fname, label='Era', preprocessor=bag_of_words): 
    df = pd.read_csv(fname)
    return (df[label].values, [preprocessor(string) for string in df['Lyrics'].values])
    

    
def oov_rate(bow1, bow2):
    return len(compute_oov(bow1, bow2)) / len(bow1.keys())
