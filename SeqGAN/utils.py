import sqlite3


class Vocab:    #Building a Vocabulary
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)                            #Create a dictionary
        self.id2word = {v: k for k, v in self.word2id.items()}  #Reverse pass dictionary to id2word
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1  #The dictionary type assignment, where .get(word,0)+1 is a count of the frequency of word occurrences

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):    #The sorted() function sorts all iterable objects, sorting them from most frequent to least frequent.
            if count < min_count:
                break
            _id = len(self.word2id)     #Current dictionary size
            self.word2id.setdefault(word, _id)  #Returns the value corresponding to the word in the dictionary, i.e. the number of occurrences of the word in the current sentence, or _id if it does not exist
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter} #Dictionary Collection {each word: corresponding id}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(path, order):

    # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)  # "City","State" ,where rownum<=10
    rows = cursor.fetchall()
    rows = [x[:-1] for x in rows]
    # print(rows)

    if order == 1:
        print("Load data in positive order……")

    elif order == 0:
        print("Loading data in reverse order……")
        rows = [x[::-1] for x in rows]
        # print(rows)
    cursor.close()
    conn.close()

    return rows

def sentence_to_ids(vocab, sentence, UNK=3):
    '''
    # Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int
    '''
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids

def pad_seq(seq, max_length, PAD=0):                    #If the length of the sentence is less than 25, 0 is added after it
    """
    :param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    '''
    :param ids: list of int,
    :param vocab:
    :param verbose(optional): 
    :return sentence: list of str
    '''
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})