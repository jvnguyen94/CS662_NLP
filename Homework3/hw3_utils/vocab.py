from collections import defaultdict
from itertools import count
import torch

BOS_SYM = '<BOS>'
EOS_SYM = '<EOS>'


def build_vocab(corpus):
    """
    Build an exhaustive character inventory for the corpus, and return dictionaries
    that can be used to map characters to indicies and vice-versa.
    
    Make sure to include BOS_SYM and EOS_SYM!
    
    :param corpus: a corpus, represented as an iterable of strings
    :returns: two dictionaries, one mapping characters to vocab indicies and another mapping indices to characters.
    :rtype: dict-like object, dict-like object
    """
    # c2i = defaultdict()
    # i2c = defaultdict()

    # for sentence in corpus:
    #     sentence = list(sentence)
    #     if BOS_SYM in c2i.keys():
    #         c2i[BOS_SYM].append(sentence[0])
    #         c2i[EOS_SYM].append(sentence[-1])
    #     else:
    #         c2i[BOS_SYM] = [sentence[0]]
    #         c2i[EOS_SYM] = [sentence[-1]]
        
    # for sentence in corpus:
    #     sentence = list(sentence)
    #     if sentence[0] in c2i.keys():
    #         c2i[sentence[0]].append(BOS_SYM)
    #     else:
    #         c2i[sentence[0]] = [BOS_SYM]
    #     if sentence[-1] in c2i.keys():
    #         c2i[sentence[-1]].append(EOS_SYM)
    #     else:
    #         c2i[sentence[-1]] = [EOS_SYM]

    #     for idx, char in enumerate(sentence):
    #         if char in c2i.keys():
    #             c2i[char].append(idx)
    #         else:
    #             c2i[char] = [idx]

    # i2c[BOS_SYM] = c2i[BOS_SYM]
    # i2c[EOS_SYM] = c2i[EOS_SYM]

    # for char in c2i.keys():
    #     if char != BOS_SYM and char != EOS_SYM:
    #         for idx in c2i[char]:
    #             if idx in i2c.keys():
    #                 i2c[idx].append(char)
    #             else:
    #                 i2c[idx] = [char]
    
    ## Get indiv characters for all sentences
    char_corpus = [list(sentence) for sentence in corpus]
    
    ## Collapse lists of lists into a singular list
    collapse_corpus = list(set([ii for sub in char_corpus for ii in sub]))
    
    ## Add BOS_SYM at beginning and EOS_SYM at the end
    collapse_corpus.insert(0, BOS_SYM)
    collapse_corpus.insert(len(collapse_corpus), EOS_SYM)

    ## create index -> char mapping
    i2c = dict(enumerate(collapse_corpus))

    ## invert to get char --> index mapping
    c2i= {value:key for key, value in i2c.items()}

    return c2i, i2c


def sentence_to_vector(s, vocab, pad_with_bos=False):
    """
    Turn a string, s, into a list of indicies in from `vocab`. 
    
    :param s: A string to turn into a vector
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: a list of the character indicies found in `s`
    :rtype: list
    """
    ## Vecterize the string into characters
    s = list(s)

    ## If wanted string padding... add BOS_SYM/EOS_SYM vars into string
    if pad_with_bos == True:
        s.insert(0, BOS_SYM)
        s.insert(len(s), EOS_SYM)

    ## Get index from vocab for character in s
    s_idx = [vocab[char] for char in s]
    

    return s_idx
    
    
def sentence_to_tensor(s, vocab, pad_with_bos=False):
    """
    :param s: A string to turn into a tensor
    :param vocab: the mapping from characters to indicies
    :param pad_with_bos: Pad the sentence with BOS_SYM/EOS_SYM markers
    :returns: (1, n) tensor where n=len(s) and the values are character indicies
    :rtype: torch.Tensor
    """
    

def build_label_vocab(labels):
    """
    Similar to build_vocab()- take a list of observed labels and return a pair of mappings to go from label to numeric index and back.
    
    The number of label indicies should be equal to the number of *distinct* labels that we see in our dataset.
    
    :param labels: a list of observed labels ("y" values)
    :returns: two dictionaries, one mapping label to indicies and the other mapping indicies to label
    :rtype: dict-like object, dict-like object
    """
