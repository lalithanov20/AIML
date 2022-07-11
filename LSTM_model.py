from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense,Input, Embedding, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Bidirectional,Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential

def train_LSTM(x,y):
    one_hot_encoded_data = pd.get_dummies(y, columns = ['Accident Level'])
    X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(x, y, test_size = 0.50, random_state = 5)
    print(X_text_test.shape)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=5000)
    #tokenizer.fit_on_texts(X_text_train)
    tokenizer.fit_on_texts(list(X_text_train.iloc[0]))
    tokenizer.fit_on_texts(list(X_text_test.iloc[0]))

    X_text_train =tokenizer.texts_to_sequences(X_text_train.iloc[0])
    X_text_test = tokenizer.texts_to_sequences(X_text_test.iloc[0])

    embedding_size = 200
    vocab_size=2000
    embeddings_dictionary = dict()

    glove_file = open('C:\\AIML\\capstone\\data\glove.6B.200d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, embedding_size))

    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    len(embeddings_dictionary.values())

    opti = Adam(lr = 0.0001)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, embeddings_initializer = Constant(embedding_matrix), 
                        input_length = 100, trainable = False))
    #embedding_layer = Embedding(vocab_size, embedding_size, weights=[embedding_matrix], trainable=False)(deep_inputs)
    model.add(Bidirectional(LSTM(128, return_sequences = True)))
    model.add(GlobalMaxPooling1D())

    model.add(Dropout(0.5, input_shape = (256,)))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5, input_shape = (128,)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5, input_shape = (64,)))
    model.add(Dense(4, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = opti, metrics = ['accuracy'])

    # Adding callbacks
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)  
    model_cp = ModelCheckpoint('accidental.h5', monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, min_lr=0.0005, verbose=1),

    # logdir = 'log'; 
    # tb = TensorBoard(logdir, histogram_freq = 1)

    # callbacks = [mcp]
    callbacks = [early_stop, model_cp, reduce_lr]

    batch_size = 100
    epochs = 20

    model.fit(np.array(X_text_train), np.array(y_text_train), epochs = epochs, 
                validation_data=(np.array(X_text_test), np.array(y_text_test)), batch_size = batch_size, verbose = 1, callbacks = callbacks)

    test_accuracy = model.evaluate(X_text_test, y_text_test, batch_size=8, verbose=0)

    return(test_accuracy)
