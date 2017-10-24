"""
Utility for vector representation of word.
This is based on TensorFlow RNN model data utils
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py

Common workflow:
* create vocabulary
* tokenize data

Files:
1. Corpus file
2. Vocabulary file
3. IDS File
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.platform import gfile

# Special vocabulary symbols
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    """
    Basic tokenizer, split the sentence into a list of tokens.
    :param sentence: sentence to tokenize.
    :return: list of token
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def initialize_vocabulary(vocabulary_file):
    """
    Initialize vocabulary from file.
    :param vocabulary_file: file containing vocabulary.
    :return: vocabulary and reversed vocabulary
    """
    if gfile.Exists(vocabulary_file):
        rev_vocab = []
        with gfile.GFile(vocabulary_file, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s doesn't exist.", vocabulary_file)


def sentence_to_token_ids(sentence, tokenizer, vocabulary):
    """
    Convert a string to list of integers representing token-ids.
    :param sentence: the sentence in bytes format to convert to token-ids.
    :param tokenizer: a function to use to tokenize each sentence.
    :param vocabulary: a dictionary mapping tokens to integers.
    :return: a list of integers, the token-ids for the sentence.
    """
    words = tokenizer(sentence)
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def tokenize_data(corpus_file, ids_file, vocabulary_file, tokenizer):
    """
    Tokenize corpus file and save it in ids file.
    :param corpus_file: source/corpus file.
    :param ids_file: target file.
    :param vocabulary_file: vocabulary file.
    :param tokenizer: tokenizer function.
    :return:
    """
    if not gfile.Exists(ids_file):
        print("Tokenize data %s" % corpus_file)
        vocab, _ = initialize_vocabulary(vocabulary_file)
        with gfile.GFile(corpus_file, mode="rb") as data_file:
            with gfile.GFile(ids_file, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("Tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line),
                                                      vocab,
                                                      tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def create_vocabulary(vocabulary_file, data_source, max_vocabulary_size, tokenizer):
    """
    Generate vocabulary from data source to vocabulary file.
    :param vocabulary_file: vocabulary file
    :param data_source: corpus file
    :param max_vocabulary_size: limit of the size of the created vocabulary
    :param tokenizer: tokenize function
    :return:
    """
    if not gfile.Exists(vocabulary_file):
        vocab = {}
        with gfile.GFile(data_source, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1

                if counter % 100000 == 0:
                    print("processing line %d" % counter)

                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w)
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_file, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")
