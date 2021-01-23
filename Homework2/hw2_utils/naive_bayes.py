from hw2_utils.constants import OFFSET
from hw2_utils import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter
from itertools import chain
import math

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """
    Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    ## Initialize counter to keep track
    counts = Counter()

    if label == None:
        for (idx, obj) in enumerate(y):
            counts.update(x[idx])

    ## Index for y to keep track of place in list
    for (idx, obj) in enumerate(y):
        ## If matches desired label... 
        if obj == label:
            ## Add the counts and save to counter
            counts.update(x[idx])   

    ## Convert to defaultdict to handle missingness
    counts = defaultdict(int, counts)

    return counts



# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    """
    Compute smoothed log-probability P(word | label) for a given label. (eq. 2.30 in Eisenstein, 4.14 in J&M)

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict 

    """
    ## Probability of word P(word | label)
    ## phi  = [smooth_hyperparam + count(y,j)] / [abs(vocab) + sigma(count(y, j'))]
   
   ## Get corpus counts for given label
    corpus_counts = get_corpus_counts(x, y, label)

    ## Sum all TOTAL NUMBER of words in the corpus
    ## Lesson: CANNOT just take the len(dict)... I had to sum(dict items)
    corpus_total_len = sum(corpus_counts.values())

    ## Get length of the vocabulary
    v = len(vocab)

    ## For each word in vocab... calculate the log phi
    word_prob =  defaultdict(float)
    for (word, _) in vocab:
        word_prob[word] = np.log(corpus_counts[word] + smoothing) - np.log((v * smoothing) + corpus_total_len)

    return word_prob
    


# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """
    Estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights, as a default dict where the keys are (label, word) tuples and values are smoothed log-probs of P(word|label)
    :rtype: defaultdict 

    """
    
    labels = set(y) 
    counts = defaultdict(float)
    doc_counts = defaultdict(float)
 
    ## Aggregregate all counts for words
    for item in x:
        counts.update(item)
    vocab = list(counts.items())
    
    ## Words per label
    label_counts = Counter(y)

    ## Iterate through each label and update the nb
    for label in labels:
        doc_counts.update(estimate_pxy(x, y, label, smoothing, vocab))
        doc_counts.update({(label, OFFSET): np.log(label_counts[label]/len(vocab))})

    return doc_counts


# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """
    Find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value, scores
    :rtype: float, dict mapping smoothing value to score
    """
  
    ## Get all unique labels
    labels = np.unique(y_tr)

    ## Keep track of top accuracy and top smoother by performance
    top_accuracy = 0
    top_smoother = None
    scores = {}

    ## Try all val in smoothers
    for val in smoothers:
        ## Estimate nb
        nb_estimate = estimate_nb(x_tr, y_tr, val)
        ## Predict on the dev set
        y_hat = clf_base.predict_all(x_dv, nb_estimate, labels)
        ## Get the accuracy
        accuracy = evaluation.acc(y_hat, y_dv)
        ## Append the score
        scores[val] = accuracy
        ## Re-write top if this iter does better
        if accuracy > top_accuracy:
            top_accuracy = accuracy
            top_smoother = val

    return top_smoother, scores
