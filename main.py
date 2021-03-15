import multiprocessing
import ssl
from datetime import datetime
from timeit import default_timer as timer

import nltk
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Embedding, LSTM, TimeDistributed, Dense
from tensorflow.python.keras.models import Sequential

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('mac_morpho')
mac_morpho = nltk.corpus.mac_morpho

TAGS_LIST = [
    'ADV-KS',
    'KS',
    'VAUX',
    'IN',
    'NPRO',
    'N',
    'PROADJ',
    'NPROP',
    'PDEN',
    'CUR',
    'ADV',
    'ADJ',
    'PCP',
    'PROSUB',
    'PREP',
    'ADV-KS-REL',
    'PROPESS',
    'PRO-KS-REL'
    'KC',
    'PROP',
    'PRO-KS',
    'ART',
    'V',
    'NUM',
    'PAD'
]


def token2word(input, dictonary):
    phrase = []
    for c in input:
        if c != 0:
            phrase.append(list(dictonary.keys())[c - 1])
    return phrase


class DataExtractor:
    def __init__(self, data, max_length=None, verbose=0):
        start = timer()
        if verbose:
            print(f'{datetime.now()} - Starting Data Extraction')
        self._data = data
        self._X, self._Y, self._word2index, self._tag2index, self._max_length = self._separate_mac_morpho_in_x_and_y()

        if verbose == 2:
            print("{} - Total number of tagged sentences: {}".format(datetime.now(), len(self._X)))
            print("{} - Vocabulary size: {}".format(datetime.now(), len(self._word2index)))
            print("{} - Total number of tags: {}".format(datetime.now(), len(self._tag2index)))
            print("{} - Biggest Sentence: {}".format(datetime.now(), self._max_length))

        self._X_encoded, self._X_tokenizer = self._tokeninze_data(self._X)
        self._Y_encoded, self._Y_tokenizer = self._tokeninze_data(self._Y)

        different_length = [1 if len(input) != len(output) else 0 for input, output in
                            zip(self._X_encoded, self._Y_encoded)]
        if verbose == 2:
            print(
                "{} - {} sentences have disparate input-output lengths.".format(datetime.now(), sum(different_length)))
        if max_length is not None:
            self._max_length = max_length
        self._X_padded, self._Y_padded = self._padding_data(self._X_encoded, max_length), self._padding_data(
            self._Y_encoded, max_length)
        self._Y_one_hot = to_categorical(self._Y_padded)
        end = timer()
        if verbose:
            print(f'{datetime.now()} - Finishing Data Extraction. Took {round(end - start, 2)}s')

    @property
    def X(self):
        return self._X

    @property
    def word_tokenizer(self):
        return self._X_tokenizer

    @property
    def tag_tokenizer(self):
        return self._Y_tokenizer

    @property
    def Y_encoded(self):
        return self._Y_encoded

    @property
    def Y_one_hot(self):
        return self._Y_one_hot

    @property
    def X_encoded(self):
        return self._X_encoded

    @property
    def Y_padded(self):
        return self._Y_padded

    @property
    def X_padded(self):
        return self._X_padded

    @property
    def Y(self):
        return self._Y

    @staticmethod
    def _padding_data(data, maxlen):
        return pad_sequences(data, maxlen=maxlen, padding="pre", truncating="post")

    @staticmethod
    def _tokeninze_data(input):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(input)
        return tokenizer.texts_to_sequences(input), tokenizer

    def _separate_mac_morpho_in_x_and_y(self):
        X, Y = [], []
        words, tags = set([]), set([])
        MAX_SEQ_LENGTH = 0
        for sentence in self._data:
            X_sentence = []
            Y_sentence = []
            count = 0
            for entity in sentence:
                word = entity[0]
                tag: str
                tag = entity[1]
                if tag.find("|") != -1:
                    tag = tag[:tag.find('|')]
                if not tag in TAGS_LIST:
                    continue

                words.add(word.lower())
                tags.add(tag)
                X_sentence.append(word)
                Y_sentence.append(tag)
                count = count + 1

            if count > MAX_SEQ_LENGTH:
                MAX_SEQ_LENGTH = len(sentence)

            if X_sentence or Y_sentence:
                X.append(X_sentence)
                Y.append(Y_sentence)

        word2index = {w: i + 1 for i, w in enumerate(list(words))}
        word2index['PAD'] = 0

        tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
        tag2index['PAD'] = 0
        return X, Y, word2index, tag2index, MAX_SEQ_LENGTH


class LSTMModel:
    def __new__(cls, vocabulary_size, embedding_size, max_seq_length, embedding_weights, num_classes):
        VOCABULARY_SIZE = vocabulary_size
        EMBEDDING_SIZE = embedding_size
        MAX_SEQ_LENGTH = max_seq_length
        embedding_weights = embedding_weights
        NUM_CLASSES = num_classes

        lstm_model = Sequential()
        lstm_model.add(
            Embedding(
                input_dim=VOCABULARY_SIZE,
                output_dim=EMBEDDING_SIZE,
                input_length=MAX_SEQ_LENGTH,
                weights=[embedding_weights],
                trainable=True
            )
        )
        lstm_model.add(LSTM(64, return_sequences=True))
        lstm_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
        lstm_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        lstm_model.summary()
        return lstm_model


class Word2VecModel:
    def __new__(cls, text, embedding_size):
        start = timer()
        print(f'{datetime.now()} - Generating Embeddings')
        cores = multiprocessing.cpu_count()
        hyperparameters = {
            "min_count": 5,
            "window": 2,
            "size": embedding_size,
            "sample": 6e-5,
            "alpha": 0.03,
            "min_alpha": 0.0007,
            "negative": 20,
            "workers": cores - 1,
            "iter": 20
        }

        model = Word2Vec(
            text,
            sg=2,
            **hyperparameters
        )
        model.save('word2vec_mac_morphos_model_2')
        end = timer()
        print(f'{datetime.now()} - Finish Generating Embeddings. Took {round(end - start, 2)}s')
        return model


def emb_weights(word2vec, word_tokenizer, vocabulary_size, embedding_size):
    VOCABULARY_SIZE = vocabulary_size
    EMBEDDING_SIZE = embedding_size
    word2id = word_tokenizer.word_index
    embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))
    for word, index in word2id.items():
        try:
            embedding_weights[index, :] = word2vec[word]
        except KeyError:
            pass
    return embedding_weights


def flat_list(non_flat_list):
    return np.array([item for sublist in non_flat_list for item in sublist])


class TrainTestAndValidation:
    def __init__(self, X, Y, test_size, valid_size):
        TEST_SIZE = test_size
        X_train, X_test, Y_train, Y_test = train_test_split(
            X,
            Y,
            test_size=TEST_SIZE,
            random_state=4
        )
        VALID_SIZE = valid_size
        X_train, X_validation, Y_train, Y_validation = train_test_split(
            X_train,
            Y_train,
            test_size=VALID_SIZE,
            random_state=4
        )
        print(f'{datetime.now()} - TRAINING DATA')
        print('Shape of input sequences: {}'.format(X_train.shape))
        print('Shape of output sequences: {}'.format(Y_train.shape))
        print("-" * 50)
        print(f'{datetime.now()} - VALIDATION DATA')
        print('Shape of input sequences: {}'.format(X_validation.shape))
        print('Shape of output sequences: {}'.format(Y_validation.shape))
        print("-" * 50)
        print(f'{datetime.now()} - TESTING DATA')
        print('Shape of input sequences: {}'.format(X_test.shape))
        print('Shape of output sequences: {}'.format(Y_test.shape))

        self._X_train, self._X_test, self._Y_train, self._Y_test = X_train, X_test, Y_train, Y_test
        self._X_validation, self._Y_validation = X_validation, Y_validation

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def X_validation(self):
        return self._X_validation

    @property
    def Y_train(self):
        return self._Y_train

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def Y_validation(self):
        return self._Y_validation


def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            max_value_index = np.argmax(categorical)
            if max_value_index != 0:
                print(max_value_index)
                token_sequence.append(index[max_value_index])
                print(token_sequence)
        token_sequences.append(token_sequence)

    return token_sequences


def main():
    # c, f = getVectorizedData()
    data_extractor = DataExtractor(mac_morpho.tagged_sents(), max_length=100, verbose=2)
    embedding_size = 130
    vocabulary_size = len(data_extractor.word_tokenizer.word_index) + 1
    try:
        word2vec = Word2Vec.load("/Users/eduardovillani/git/nlp-tp2/word2vec_mac_morphos_model_2")
    except Exception:
        word2vec = Word2VecModel(flat_list(data_extractor.X), embedding_size)

    lstm = LSTMModel(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        max_seq_length=100,
        embedding_weights=emb_weights(
            word2vec=word2vec,
            word_tokenizer=data_extractor.word_tokenizer,
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size
        ),
        num_classes=data_extractor.Y_one_hot.shape[2]
    )

    train_test_and_validation = TrainTestAndValidation(data_extractor.X_padded, data_extractor.Y_one_hot, 0.2, 0.2)
    lstm.fit(
        train_test_and_validation.X_train,
        train_test_and_validation.Y_train,
        batch_size=128,
        epochs=8,
        validation_data=(train_test_and_validation.X_validation, train_test_and_validation.Y_validation)
    )
    lstm.save('lstm_v1_embbedings')
    from keras.models import load_model

    model_lstm = load_model("lstm_v1_embbedings")
    lstm.evaluate(
        train_test_and_validation.X_test, train_test_and_validation.Y_test, verbose=1
    )

    # pred = model_lstm.predict(
    #         train_test_and_validation.X_test
    #     )


if __name__ == '__main__':
    main()
