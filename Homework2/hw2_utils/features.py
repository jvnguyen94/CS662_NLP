from  hw2_utils import constants 
import numpy as np
from math import floor
# deliverable 4.1
def get_token_type_ratio(counts):
    """
    Compute the ratio of tokens to types
    
    :param counts: bag of words feature for a song
    :returns: ratio of tokens to types
    :rtype float
    """

    ## ratio = all counts / unique words
    ratio = 0
    ## Make sure that counts has data... otherwise return 0
    if len(counts): 
        ratio = sum(counts.values()) / len(counts)

    return ratio



# deliverable 4.2
def concat_ttr_binned_features(data):
    """
    Add binned token-type ratio features to the observation represented by data
    
    :param data: Bag of words
    :returns: Bag of words, plus binned ttr features
    :rtype: dict
    """
    ## Calculate ttr
    ttr = get_token_type_ratio(data)

    ## Create int bins to represent the different variables 
    bins = [0, 1, 2, 3, 4, 5, 6]
    
    ## Create list for var names and a list of zeros to keep track 
    keys = [constants.TTR_ZERO, constants.TTR_ONE, constants.TTR_TWO, constants.TTR_THREE, constants.TTR_FOUR, constants.TTR_FIVE, constants.TTR_SIX]
    values = np.zeros(7, dtype=int)

    ## Iter through bin and check if floor(ttr) matches bin
    ## If match - then flag idx to change value at idx
    for (idx,item) in enumerate(bins):
        check = floor(ttr) - item
        if check == 0:
            values[idx] = 1
    ## If not flag for first 6 bins - then flag last infinity bin
    if sum(values) == 0:
        values[6] = 1            

    ## Zip two lists into dictionary
    ttr_features = {keys[i]: values[i] for i in range(len(keys))}
  
    ## Add features to data
    data.update(ttr_features)
    
    return data
