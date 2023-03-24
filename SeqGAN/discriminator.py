import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers import Dense, Embedding, LSTM
from SeqGAN.highway import Highway


def Discriminator(V, E, H=64, dropout=0.1):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator