try:
    import nltk
except:
    k=0
import urllib
import cPickle as pickle
import numpy as np
from nltk.corpus import cmudict
from poemCorpusTrainer.CorpusLoader import *
from poemCorpusTrainer.Rhymer import *
from poemCorpusTrainer.POSTrainer import *
import copy

if __name__ == "__main__":
    """Uncomment these lines to train on new corpus"""
    #corpus_loader = CorpusLoader('poem_corpus.txt', populate_phonemes=True)
    #pickle.dump(corpus_loader, open('corpus_loader_01.cl', 'wb'))
    corpus_loader = pickle.load(open('corpus_loader_01.cl', 'rb'))
    rhymer = Rhymer(corpus_loader)

    pos_corpus_loader = CorpusLoader('pos_corpus.txt', populate_phonemes=False)
    pos_trainer = POSTrainer(pos_corpus_loader)
    
    txt = raw_input('Please enter the sentence to be rhymed-\n')
    print '\nRhyme suggestions-\n', rhymer.find_rhymes(txt)

    k = raw_input()