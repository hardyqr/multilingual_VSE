# Fangyu @ EPFL
# June 29, 2018

import nltk
import pickle
import argparse
from collections import Counter
import pandas as pd
import sys
from tqdm import tqdm
from text_preprocessing import TextPreprocessing # Rémi's lib
from flair.models import SequenceTagger
from flair.data import Sentence


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word 
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def tagger_and_preprocessor(text, tags, tagger, preprocess):
    """
    Args:
        text: a string.
        tags: a list of string.
        tagger: flair tagger.
        preprocess: tokenizer from torch-goodies.
    return:
        a lsit of tagged tokens.
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
    sentence_tagged = sentence.to_tagged_string()
    tokens_tagged = sentence_tagged.split(' ')
    tmp = []
    tokens = []
    for w in tokens_tagged:
        if w not in tags:
            tmp += [w]
        else:
            tokens += preprocess(" ".join(tmp))
            tokens += [w]
            tmp = []
    if len(tmp) > 0: 
        tokens += preprocess(" ".join(tmp))
    return tokens


def _build_vocab(csv_file_path, threshold, lang):
    """ Build a simple vocabulary wrapper. """
    """ used for buidling vocab with NER """
    """
    This function achieves two things: 
        - build vocab
        - tokenize & tag text, then save it as a string seprated by ' '

    return:
        - vocab
        - new dataframe with a new field that contains tokenize & tag text.
    """
    
    # define preprocessor
    preprocess = TextPreprocessing(
            twitter=False, replace_number=False, clean_html=True,
            capitalize=True, repeat=True, elong=True,
            replace_url=True, replace_emoticon=True,
            lower=True, lang=lang)
    tagger = SequenceTagger.load('de-ner')

    print ('opening caption file...')
    df_data = None
    with open(csv_file_path,'rb') as f:
        df_data = pd.read_csv(f)
    print ('caption file read in.')
    # the csv is like ['id','img_name',,'caption']

    tags = ['<B-PER>', '<E-PER>', '<S-ORG>', '<S-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<E-ORG>', '<S-LOC>', '<B-LOC>', '<E-LOC>', '<S-MISC>', '<I-LOC>', '<B-MISC>', '<E-MISC>', '<I-MISC>']
    counter = Counter()
    df_data['caption_token_list'] = '<to-be-replaced>'
    # add a column to dataframe
    for index, row in df_data.iterrows():
        caption = str(row.caption)
        
        # replace nltk with torch-goodies
        #tokens = nltk.tokenize.word_tokenize(caption.lower())

        tokens = tagger_and_preprocessor(caption, tags, tagger, preprocess)
        # add tokenized & tagged list to df
        df_data.at[index,'caption_token_list'] = " ".join(tokens)
        counter.update(tokens)

        if (index+1) % 1000 == 0:#
            print("[{}/{}] Tokenized the captions.".format(index+1, len(df_data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    #print (words)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab, df_data



def build_vocab(csv_file_path, threshold, lang):
    """ Build a simple vocabulary wrapper. """
    """
    This function achieves two things: 
        - build vocab
        - tokenize & tag text, then save it as a string seprated by ' '

    return:
        - vocab
        - new dataframe with a new field that contains tokenize & tag text.
    """
    
    print ('opening caption file...')
    #df_data = None
    with open(csv_file_path,'rb') as f:
        df_data = pd.read_csv(f)
    print ('caption file read in.')

    counter = Counter()
    # add a column to dataframe
    for index, row in enumerate(df_data.itertuples()):
        cap = str(row.caption).split(" ")
        title = str(row.title).split(" ")
        lead = str(row.lead).split(" ")
        article = str(row.article).split(" ")

        tokens = cap + title + lead + article  # being preprocessed before

        counter.update(tokens)

        if (index+1) % 1000 == 0:#
            print("[{}/{}] Tokenized the captions.".format(index+1, len(df_data)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    #print (words)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab




def main(args):
    vocab = build_vocab(csv_file_path=args.caption_path, threshold=args.threshold,lang=args.language)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='data/some_file.csv', 
                        help='path for train annotation file')
    parser.add_argument('--out_caption_path', type=str, 
                        default='some_file.csv', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./vocab/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    parser.add_argument('--language', type=str, default='de', 
                        help='choose language: en, de, fr, it')
    args = parser.parse_args()
    main(args)
