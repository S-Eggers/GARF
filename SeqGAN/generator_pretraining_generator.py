import numpy as np
import random
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from SeqGAN.utils import load_data, Vocab, pad_seq, sentence_to_ids


class GeneratorPretrainingGenerator(Sequence):              #Take the data directly from the original data and make x and y_true as x_train and y_train i.e. training data and labels
    def __init__(self, path, order, B, T, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path = path
        self.B = B
        self.T = T
        self.min_count = min_count
        self.count = 0

        sentences = load_data(path, order)

        print("Raw data", sentences)
        self.rows = sentences

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
        # with open(path, 'r', encoding='utf-8') as f:
        #     self.n_data = sum(1 for line in f)          #Number of data rows
        self.n_data = len(sentences)                  # Number of original data rows
        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        f = open('data/save/word2id.txt', 'w')
        f.write(str(self.word2id))
        f.close()
        f = open('data/save/id2word.txt', 'w')
        f.write(str(self.id2word))
        f.close()
        print("+++++++")
        #record f.read().lower() is case-insensitive; chars = sorted(list(set(raw_text))) can be split into character-level sorting, where set is de-weighted and sorted
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        
        self.reset()


    def __len__(self):
        return self.n_data // self.B        #Total number of data divided by the number of samples selected for one training

    def __getitem__(self, idx):             #Read the idx row of the original data, generate x and y_true
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, max_length)
            y_true: numpy.array, shape = (B, max_length, V)
                labels with one-hot encoding.
                max_length is the max length of sequence in the batch.
                if length smaller than max_length, the data will be padded.
        '''
        # print("****************************")
        # self.count =self.count+1
        # print(self.count)
        # print("idx is",idx)

        x, y_true = [], []
        start = (idx-1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                # print("self.shuffle:",self.shuffle)
                idx = self.shuffled_indices[i]
            else:
                # print("shuffle=False")
                idx = i
            # sentence = linecache.getline(self.path, idx)    #Read the idx row of the original data
            # words = sentence.strip().split()
            # print(idx)
            sentence = self.rows[idx]                         #Read the idx row of the query result
            words = []
            for i in sentence:
                words.append(i)
            ids = sentence_to_ids(self.vocab, words)        #ids is a list that holds the sequence of word ids in the original data

            ids_x, ids_y_true = [], []                      #place empty

            ids_x.append(self.BOS)                          #Begin by writing the identifier BOS
            ids_x.extend(ids)                               #Add ids, i.e., the ids corresponding to the multiple words in the current sentence
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)                                 #X is a list of multiple sentences combined,np.array(x)).shape=(B,), i.e. B rows, 1 element per row
            # print("x:",x)
            # print(type(x))
            # print((np.array(x)).shape)

            ids_y_true.extend(ids)
            ids_y_true.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)                       #As of now, y_true and x, with similar data and shape, are both (B,), except that each element in them has one less beginning BOS
            # print("y_true:",y_true)
            # print("y_true:")
            # print(type(y_true))
            # print((np.array(y_true)).shape)


            max_length = max(max_length, len(ids_x))


        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]                #loop len(X) times, X[i] is the i-th sentence in the list X, and truncated to the length of max_length

        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]     #If the length of the sentence is less than 25, 0 is added after it
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        # print("y_true:", y_true[0][0])
        y_true = to_categorical(y_true, num_classes=self.V)     #The original category vector is converted to one-hot form, and the dimension is the total number of words.
        # print("x:", x[0])
        # print("y_true after conversion:",y_true[0][0])
        # print("x:", x)
        # print("y_true:",y_true)

        return (x, y_true)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):                                    #Reset to regenerate a jumbled array of size n_data
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)           #Break up the elements in the list
        #print(self.shuffled_indices)                       #A scrambled array of size n_data, [3850 1111 4587 ... 2454 3013 3144]

    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self