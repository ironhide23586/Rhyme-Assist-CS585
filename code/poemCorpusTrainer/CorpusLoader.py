import numpy as np
import nltk
from nltk.corpus import cmudict
import re

class CorpusLoader(object):
    """Encapsulates functionality to process poetic corpus"""

    """List of strings where each element is a poetic line"""
    corpus = None

    """List of POS tags corresponding to each poetic line in corpus."""
    corpus_tags = None
    
    """List of phonetic disambiguations pertaining to each line in poetic corpus."""
    corpus_phonemes = None

    def __init__(self, file_name=None, populate_phonemes= True):
        if file_name is not None:
            self.load_corpus(file_name, populate_phonemes=populate_phonemes)

    def load_corpus(self, file_name, populate_phonemes=True):
        f = open(file_name, 'rb')
        self.corpus = []
        for line in f:
            l = line.strip()
            if l != '':
                self.corpus.append(l.split())
        f.close()
        self.corpus = np.array(self.corpus)
        self.corpus_tags = np.array([[elem[1] for elem in nltk.pos_tag(corpus_elem)] for corpus_elem in self.corpus])
        self.corpus_phonemes = np.array([None] * self.corpus.shape[0])
        i = 0
        if populate_phonemes == True:
            for sen in self.corpus:
                print i
                try:
                    self.corpus_phonemes[i] = cmudict.dict()[re.sub('[^a-zA-Z]', ' ', sen[-1]).lower().strip().split()[-1]]
                except:
                    tmp = 0
                i += 1