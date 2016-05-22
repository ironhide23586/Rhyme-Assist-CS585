from poemCorpusTrainer.CorpusLoader import *
import copy

class Rhymer(object):
    """Performs rhyme generation."""

    corpus_loader = None

    def __init__(self, cp_loader):
        self.corpus_loader = cp_loader

    def phoneme_sim(self, p_wrd, q_wrd):
        if p_wrd[-1] != q_wrd[-1]:
            return 0
        i = 0
        score = 0
        p = copy.deepcopy(p_wrd)
        q = copy.deepcopy(q_wrd)
        p.reverse()
        q.reverse()
        for p_elem in p:
            if i == len(q):
                break
            if p_elem in q[i:]:
                score += 1
            i += 1
        return 1. * score / len(p)


    def find_rhymes(self, sen):
        wrd = re.sub('[^a-zA-Z]', ' ', sen.split()[-1]).lower().strip().split()[-1]
        wrd_ph = None
        try:
            wrd_ph = cmudict.dict()[wrd]
        except:
            return None
        idx = []
        for i in xrange(self.corpus_loader.corpus.shape[0]):
            sims = []
            c_ph = self.corpus_loader.corpus_phonemes[i]
            if c_ph is None:
                continue
            for wp in wrd_ph:
                for cp in c_ph:
                    sims.append(self.phoneme_sim(wp, cp))
            if max(sims) > .5:
                idx.append(i)
        return [' '.join(e) for e in self.corpus_loader.corpus[idx]]