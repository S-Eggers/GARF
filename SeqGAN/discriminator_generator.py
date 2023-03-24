import numpy as np
import random
from keras.utils import Sequence
from SeqGAN.utils import load_data, Vocab, pad_seq, sentence_to_ids
import linecache


class DiscriminatorGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        X, Y = generator.__getitem__(idx=11)
        print(X[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(Y)
        >>> 0, 1, 1, 0, 1, 0, 0, ..., 1

        id2word = generator.id2word

        x_words = [id2word[id] for id in X[0]]
        print(x_words)
        >>> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path_pos, order, path_neg, B, T=40, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T
        self.min_count = min_count

        sentences = load_data(path_pos, order)
        self.rows=sentences

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)

        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)


        # with open(path_pos, 'r', encoding='utf-8') as f:
        #     self.n_data_pos = sum(1 for line in f)              #Number of original data rows

        self.n_data_pos = len(self.rows)                             #Number of original data rows
        with open(path_neg, 'r', encoding='utf-8') as f:
            self.n_data_neg = sum(1 for line in f)              #Number of rows of generated data
        # f = open('data/save/word2id-d.txt', 'w')
        # f.write(str(self.word2id))
        # f.close()

        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            X: numpy.array, shape = (B, max_length)
            Y: numpy.array, shape = (B, ) ,label:true=1,generated data=0
        '''
        X, Y = [], []
        start = (idx-1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0

        for i in range(start, end):
            # print(start)
            # print(end)
            # print("Ex：",idx)
            idx = self.indicies[i]    #Randomly select a value from the original data and the generated data index
            # print("After",idx)
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                # sentence = linecache.getline(self.path_pos, idx) # str  #Read the idx row of the original data
                sentence = self.rows[idx]
                words = []
                for i in sentence:
                    words.append(i)
            elif is_pos == 0:
                sentence = linecache.getline(self.path_neg, idx) # str  # Read the idx row of the generated data
                words = sentence.strip().split()
            # words = sentence.strip().split()  # list of str  ex.['"261318"', '"SALEM"', '"MO"', '"65560"', '"DENT"', '"5737296626"', '"Pregnancy', 'and', 'Delivery', 'Care"', '"PC_01"', '"Elective', 'Delivery"']
            # print("word:",words)
            ids = sentence_to_ids(self.vocab, words) # list of ids ex.[1261, 1262, 51, 1263, 1264, 1265, 136, 31, 137, 27, 138, 139, 140]
            # print("ids:",ids)

            x = []
            x.extend(ids)
            x.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            X.append(x)                             #句子合集，ex.[[703, 250, 52, 704, 250, 705, 71, 27, 72, 73, 74, 8, 75, 76, 31, 77, 78, 2], [421, 422, 9, 423, 231, 424, 42, 27, 89, 90, 91, 92, 93, 94, 2]]
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]                #Remove the part that exceeds the maximum length

        X = [pad_seq(sen, max_length) for sen in X] #The end of the current part to the maximum length part of the complement 0
        X = np.array(X, dtype=np.int32)
        # print("X:",X)

        return (X, Y)

    def next(self):
        # print(self.idx)
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        # print(X)
        # print(Y)
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)        #Get an array starting from 1, with the size of the original data rows ex. [1,2,3]
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)   #Get an array starting from -1 with the size of the number of rows of generated data ex. [-1,-2,-3,-4]
        self.indicies = np.concatenate([pos_indices, neg_indices])      #Link,ex. [1,2,3,-1,-2,-3,-4]
        # print(pos_indices)                                              #In this example is [ 1 2 3 ... 5344 5345 5346] length is the number of rows of the original data
        # print(neg_indices)                                              #In this case it is [-1 -2 ...... -500] Length is the number of rows of generated data
        if self.shuffle:
            random.shuffle(self.indicies)                               #disordered [-1... -500 1..n_data_pos]
    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self
