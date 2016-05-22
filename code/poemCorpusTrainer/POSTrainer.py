from NNPackage import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class POSTrainer(object):
    """Trains neural network for identifying rhymes having related semantic sense."""

    corpus_loader = None

    tags = np.array(['$', "''", '(', ')', ',', '--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', '-NONE-'])

    clf = None

    tag_idx_map = None

    def __init__(self, cp_loader):
        self.corpus_loader = cp_loader
        self.tag_idx_map = dict().fromkeys(self.tags)
        i = 0
        for t in self.tags:
            self.tag_idx_map[t] = i
            i += 1
        self.clf = NeuralNet(self.tags.shape[0], self.tags.shape[0])

        x_train, y_train = self.gen_nn_data(self.corpus_loader.corpus_tags[:24])
        x_test, y_test = self.gen_nn_data(self.corpus_loader.corpus_tags[24:])

        self.clf.load_train_data(x_train, y_train)
        

    def train(self):
        while True:
            clf.train()
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print acc
            p, r, f, s = precision_recall_fscore_support(y_test, y_pred)
            print p
            print r
            print f
            lol=0

    def gen_nn_data(self, word_tags):
        x_word_tags = word_tags[np.arange(word_tags.shape[0])[np.arange(word_tags.shape[0]) % 2 == 0]]
        y_word_tags = word_tags[np.arange(word_tags.shape[0])[np.arange(word_tags.shape[0]) % 2 == 1]]
        x_num = self.convert_features_to_numeric(x_word_tags)
        y_num = self.convert_features_to_numeric(y_word_tags)
        return x_num, y_num

    def convert_features_to_numeric(self, word_tags):
        return np.array([self.convert_to_feature(word_tags_elem) for word_tags_elem in word_tags])

    def convert_to_feature(self, pos_list):
        feature = np.zeros(self.tags.shape[0])
        feature[[self.tag_idx_map[l_elem] for l_elem in list(set(pos_list))]] = 1
        return feature