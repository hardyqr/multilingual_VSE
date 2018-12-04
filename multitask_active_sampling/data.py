import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import argparse
import io
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import pandas as pd
import sentencepiece as spm
import fastText
from text_preprocessing import TextPreprocessing
from embeddings import FastTextVocab
from embeddings import FastTextEmbeddings
from build_vocab import * # tagger_and_preprocessor
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# flair
from flair.data import Sentence

preprocess = TextPreprocessing(
        twitter=False, replace_number=False, clean_html=True,
        capitalize=True, repeat=True, elong=True,
        replace_url=True, replace_emoticon=True,
        lower=True, lang="de")

#ft_vocab = pickle.load(open('./vocab/ft_vocab_min4_max8.pkl','rb'))

#tagger = SequenceTagger.load('de-ner')

#tags = ['<B-PER>', '<E-PER>', '<S-ORG>', '<S-PER>', '<I-PER>', '<B-ORG>', '<I-ORG>', '<E-ORG>', '<S-LOC>', '<B-LOC>', '<E-LOC>', '<S-MISC>', '<I-LOC>', '<B-MISC>', '<E-MISC>', '<I-MISC>']

#sp = spm.SentencePieceProcessor()
#sp.Load('/home/fangyu/data/de.wiki.bpe.op100000.model')

def get_paths(path, name='coco', use_restval=False):
    """
    Returns paths to images and annotations for the given datasets. For MSCOCO
    indices are also returned to control the data split being used.
    The indices are extracted from the Karpathy et al. splits using this
    snippet:

    >>> import json
    >>> dataset=json.load(open('dataset_coco.json','r'))
    >>> A=[]
    >>> for i in range(len(D['images'])):
    ...   if D['images'][i]['split'] == 'val':
    ...     A+=D['images'][i]['sentids'][:5]
    ...

    :param name: Dataset names
    :param use_restval: If True, the the `restval` data is included in train.
    """
    roots = {}
    ids = {}
    if 'coco' == name:
        imgdir = os.path.join(path, 'images')
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'article_db' == name:
        #roots['img'] =
        #roots['cap'] =
        roots = None
        ids = None
    return roots, ids

     
def bpe(text,sp):
    tokens = sp.EncodeAsPieces(text)
    tokens = [token.decode('UTF-8') for token in tokens]
    try:
        tokens[0] = tokens[0][1:] # sometimes this fails
    except:
        tokens = ['']
    return tokens


def numericalize(s,vocab,tokenized=False,_bpe=True):
    """
    Args:
        s: a string
    return:
        A torch.Tensor
    """
    tokens = None
    if tokenized:
        tokens = str(s).split(" ")
    else:
        tokens = preprocess(str(s))
    #tokens = tagger_and_preprocessor(str(s),tags,tagger,preprocess) # replace this line with the one above to go back to the untagged mode
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    """ # for bpe subword
    if s == '':  # if empty, then no bpe
        caption.append(vocab(s))
        caption.append(vocab('<end>'))
        return  torch.Tensor(caption)

    # add subwords
    for token in tokens:
        if vocab(token) == vocab.word2idx['<unk>'] and _bpe is True:
            subwords = bpe(token,sp)
            caption.extend([vocab(subword) for subword in subwords])
        else:
            caption.append(vocab(token))
    #caption.extend([vocab(token) for token in tokens]) # original
    caption.append(vocab('<end>'))
    """
    return  torch.Tensor(caption)

def _padder_(tokens,seqlen):
    if len(tokens) > seqlen:
        return tokens[:seqlen]
    else:
        return tokens + (seqlen-len(tokens))*[""]




class ArticleDatasetCSE(data.Dataset):
    """article_db Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, csv_path, vocab, transform=None, ids=None,label='caption',random_miss=True,only_test=None):
        """
        Args:
            root: image directory.
            csv_path: article_db annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
            random_miss: radomly substitue text sources with zeros (robust to missing input).
        """
        self.root = root
        self.csv_path = csv_path
        self.csv_df = pd.read_csv(open(self.csv_path,'rb'),index_col=False)
        self.ids = self.csv_df['id'].tolist()
        self.vocab = vocab
        self.transform = transform
        self.label = label
        self.random_miss = random_miss
        self.only_test = only_test
        
        
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        label = self.label
        root, text, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        seqlen = None
        if label == "article":
            text = text[:3000] # use first 3k chars
        else:
            text = text[:300] # use first 300 chars

        sentence = Sentence(text)

        return image,sentence,index,img_id

    def get_raw_item(self, index):
        row = self.csv_df.iloc[index]
        #print (row)
        pair_id = row[['id']].tolist()[0]
        path = row[['img_name']].tolist()[0]
        caption = row[[self.label]].tolist()[0]
                
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        return self.root, caption, pair_id, path, image #, bboxes
        #return self.root, caption, pair_id, path, image, subpics

    def __len__(self):
        return len(self.ids)


class ArticleDatasetUnimodal(data.Dataset):
    """article_db Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, csv_path, vocab_de, vocab_fr, transform=None, ids=None,label='caption',random_miss=True,only_test=None,train=True):
        """
        Args:
            root: image directory.
            csv_path: article_db annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
            random_miss: radomly substitue text sources with zeros (robust to missing input).
        """
        self.root = root
        self.csv_path = csv_path
        self.csv_df = pd.read_csv(open(self.csv_path,'rb'),index_col=False)

        self.ids = self.csv_df['id'].tolist()
        self.vocab_de = vocab_de
        self.vocab_fr = vocab_fr
        self.transform = transform
        self.label = label
        self.random_miss = random_miss
        self.only_test = only_test
        self.langid = fastText.load_model('/mnt/storage01/fangyu/fastText/lid.176.bin')
        print ('[data.py - inited!]')
        
        
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab_de = self.vocab_de
        vocab_fr = self.vocab_fr
        label = self.label
        #root, caption, img_id, path, image, subpics = self.get_raw_item(index)
        root, text, img_id, img_name, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
            #for i,pic in enumerate(subpics):
            #    subpics[i] = self.transform(pic)
        
        lang, prob = self.langid.predict(text.split("\n")[0])
        lang = lang[0][-2:]
        lang_num = 0
        if lang == "fr": lang_num = 1
        text_tok = text.split(" ") # if preprocessed
        #text_tok = preprocess(text) # if not preprocessed
        seqlen = None
        if label == "caption": seqlen = 40
        elif label == "article": seqlen = 150
        elif label == "title": seqlen = 20
        elif label == "lead": seqlen = 70
        text_tok_padded = _padder_(text_tok,seqlen)
        #print (text_tok_padded)
        fname = 'toks.csv'
        #fout = io.open(fname, 'a', encoding='utf-8')
        #for t in text_tok_padded:
        #    if t == "\n":
        #        print ("shit!!")
        #fout.write(" ".join(text_tok_padded[:20])+ " <END>\n")
        #fout.close()
        text_i, text_o = None, None
        # numericalization
        if lang_num == 0:
            text_i,text_o = vocab_de.numericalize([text_tok_padded])[0]
        else:
            text_i,text_o = vocab_fr.numericalize([text_tok_padded])[0]
        
        text_i = torch.LongTensor(text_i)
        text_o = torch.LongTensor(text_o)
       
        # to indicate language
        # 0 - de, 1 - fr
        #print (text_i.size())
        text_i = torch.cat((torch.LongTensor([lang_num]),text_i)) 
        #print (text_i.size())
        #text_i = text[0]
        #text_o = text[1]

        return image,text_i,text_o,index,img_id
        #return image, subpics, target[0], target[1], target[2], target[3], index, img_id
        #return image, target[0],target[1],target[2],target[3],index, img_id
        #return image, input_target[0], offset_target[0], input_target[1], offset_target[1], input_target[2], offset_target[2], input_target[3]), offset_target[3], index, img_id


    def get_raw_item(self, index):
        row = self.csv_df.iloc[index]
        #print (row)
        pair_id = row[['id']].tolist()[0]
        img_name = row[['img_name']].tolist()[0]
        text = row[[self.label]].tolist()[0]
        
        # use precomputed indices&offsets
        #text = (torch.LongTensor([0]),torch.LongTensor([0]))
        #text = self.texts[img_name]
                
        image = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        return self.root, text, pair_id, img_name, image #, bboxes
        #return self.root, caption, pair_id, path, image, subpics

    def __len__(self):
        return len(self.ids)



class ArticleDataset(data.Dataset):
    """article_db Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, csv_path, vocab, transform=None, ids=None,label='caption',random_miss=True,only_test=None):
        """
        Args:
            root: image directory.
            csv_path: article_db annotation file path.
            vocab: vocabulary wrapper.
            transform: transformer for image.
            random_miss: radomly substitue text sources with zeros (robust to missing input).
        """
        self.root = root
        self.csv_path = csv_path
        self.csv_df = pd.read_csv(open(self.csv_path,'rb'),index_col=False)
        self.ids = self.csv_df['id'].tolist()
        self.vocab = vocab
        self.transform = transform
        self.label = label
        self.random_miss = random_miss
        self.only_test = only_test
        
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        #root, caption, img_id, path, image, subpics = self.get_raw_item(index)
        root, texts, img_id, path, image = self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)
            #for i,pic in enumerate(subpics):
            #    subpics[i] = self.transform(pic)


        # Convert caption (string) to word ids.
        # (caption, article, title, lead)
        """
        target = [numericalize(caption[0],vocab),
                    numericalize(caption[1],vocab,_bpe=False),
                    numericalize(caption[2],vocab),
                    numericalize(caption[3],vocab,_bpe=False)
                    ]
        """


        #cap = preprocess(caption[0])
        #art = preprocess(caption[1])
        #tit = preprocess(caption[2])
        #led = preprocess(caption[3])
        cap = texts[0].split(" ")
        art = texts[1].split(" ")
        tit = texts[2].split(" ")
        led = texts[3].split(" ")


        target = [cap,art,tit,led]
        #target = [cap,art]

        # model robust to less input sources
        if self.random_miss:
            not_missing_index = np.random.randint(4)
            for i in range(4):
                if i == not_missing_index: continue
                # 0.3 prob. miss some input
                if np.random.randint(10) >= 7:
                    target[i] = preprocess(" ")

        # pad cap, art, tit, lead
        cap = _padder_(target[0],70)
        art = _padder_(target[1],150)
        tit = _padder_(target[2],20)
        led = _padder_(target[3],70)
        target = [cap,art,tit,led]
        #target = [cap,art]

        '''
        # for val phase, only test one input
        if self.only_test == 'article':
            target[0] = numericalize('',vocab)
            target[2] = numericalize('',vocab)
            target[3] = numericalize('',vocab)
        elif self.only_test == 'caption':
            target[1] = numericalize('',vocab)
            target[2] = numericalize('',vocab)
            target[3] = numericalize('',vocab)
        elif self.only_test == 'title':
            target[0] = numericalize('',vocab)
            target[1] = numericalize('',vocab)
            target[3] = numericalize('',vocab)
        elif self.only_test == 'lead':
            target[0] = numericalize('',vocab)
            target[1] = numericalize('',vocab)
            target[2] = numericalize('',vocab)
        '''
        # numericalization
        # every tuple (,) is for a single word
        cap_i,cap_o = vocab.numericalize([target[0]])[0]
        art_i,art_o = vocab.numericalize([target[1]])[0]
        tit_i,tit_o = vocab.numericalize([target[2]])[0]
        led_i,led_o = vocab.numericalize([target[3]])[0]
      
        cap_i = torch.LongTensor(cap_i)
        cap_o = torch.LongTensor(cap_o)
        art_i = torch.LongTensor(art_i)
        art_o = torch.LongTensor(art_o)
        tit_i = torch.LongTensor(tit_i)
        tit_o = torch.LongTensor(tit_o)
        led_i = torch.LongTensor(led_i)
        led_o = torch.LongTensor(led_o)

        return image,cap_i,cap_o,art_i,art_o,tit_i,tit_o,led_i,led_o,index,img_id
        #return image,cap_i, cap_o, art_i, art_o, index,img_id
        #return image, subpics, target[0], target[1], target[2], target[3], index, img_id
        #return image, target[0],target[1],target[2],target[3],index, img_id
        #return image, input_target[0], offset_target[0], input_target[1], offset_target[1], input_target[2], offset_target[2], input_target[3]), offset_target[3], index, img_id


    def get_raw_item(self, index):
        row = self.csv_df.iloc[index]
        #print (row)
        pair_id = row[['id']].tolist()[0]
        path = row[['img_name']].tolist()[0]
        texts = [
                row[['caption']].tolist()[0],
                row[['article']].tolist()[0],
                row[['title']].tolist()[0],
                row[['lead']].tolist()[0]
                ]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # also get small images
        

        return self.root, texts, pair_id, path, image #, bboxes
        #return self.root, caption, pair_id, path, image, subpics

    def __len__(self):
        return len(self.ids)


def padder(l_m,seqlen):
    """
    Args:
        l_m: A list of matrices (captions)
    """
    lengths = [len(m) if len(m) <= seqlen else seqlen for m in l_m]
    targets = torch.zeros(len(l_m), max(lengths)).long()
    #targets = torch.zeros(len(captions), seqlen).long() # set 128 as the cuttoff for a sentence
    for i, cap in enumerate(l_m):
        end = lengths[i]
        # avoid too long caption
        if end <= seqlen:
            targets[i, :end] = torch.tensor(cap[:end])
        else:
            targets[i, :seqlen] = cap[:seqlen]
        '''# for 'seqlen' uniform length
        if end <= seqlen:
            #targets[i,(seqlen-end):seqlen] = cap[:end] # pad from the beginning
            targets[i,:end] = cap[:end]
        else:
            targets[i,:seqlen] = cap[:seqlen] 
        lengths[i] = seqlen
        '''
    return targets, lengths

def _padder(l_t,seqlen):
    """
    """
    l_t = [row if len(row) <= seqlen else row[:seqlen] for row in l_t]
    lengths = [len(m) for m in l_t]
    max_len = max(lengths)
    for i,row in enumerate(l_t):
        if len(row) < max_len:
            l_t[i] += (max_len-len(row))*[""]
    return l_t


def collate_fn_cse(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Works with EncoderTextCSE.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images,texts,ids,img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #images, captions,articles,titles,leads,ids, img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #_,_,articles,_,_ = zip(*data_copy) # for different ordering of 

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    return images,texts,ids

def collate_fn_unimodal(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Works with EncoderTextUnimodal.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    #data.sort(key=lambda x: len(x[1]), reverse=True)
    #data_copy = data
    #try:
    #    data_copy.sort(key=lambda x: len(x[2]), reverse=True)
    #except:
    #    print ('every thing is fine.')
    images,texts_i,texts_o,ids,img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #images, captions,articles,titles,leads,ids, img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #_,_,articles,_,_ = zip(*data_copy) # for different ordering of 

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    targets = [texts_i,texts_o]

    return images,targets,ids



def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    #data.sort(key=lambda x: len(x[1]), reverse=True)
    #data_copy = data
    #try:
    #    data_copy.sort(key=lambda x: len(x[2]), reverse=True)
    #except:
    #    print ('every thing is fine.')
    images,caps_i,caps_o,arts_i,arts_o,tits_i,tits_o,leds_i,leds_o,ids,img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #images, caps_i,caps_o, arts_i,arts_o, ids, img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)

    #images, captions,articles,titles,leads,ids, img_ids = zip(*data) # img, captions, index, img_ids (ori in csv)
    #_,_,articles,_,_ = zip(*data_copy) # for different ordering of 

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    #batch_size = len(images)
    #if captions[0] is None: captions = [[0]]*batch_size
    #if articles[0] is None: articles = [[0]]*batch_size
    #if titles[0] is None: titles = [[0]]*batch_size
    #if leads[0] is None: leads = [[0]]*batch_size

    #targets_cap,lengths_cap = padder(captions,cap_seqlen)
    #targets_art,lengths_art = padder(articles,art_seqlen)
    #targets_title,lengths_title = padder(titles,cap_seqlen)
    #targets_lead,lengths_lead = padder(leads,cap_seqlen)
    #targets = [targets_cap,targets_art,targets_title,targets_lead]
    #lengths = [lengths_cap,lengths_art,lengths_title,lengths_lead]

    #caps = _padder(caps,cap_seqlen)
    #arts = _padder(arts,art_seqlen)
    #tits = _padder(tits,cap_seqlen)
    #leds = _padder(leds,cap_seqlen)
    #articles,lengths_art = padder(articles,art_seqlen)
    #titles,lengths_title = padder(titles,cap_seqlen)
    #leads,lengths_lead = padder(leads,cap_seqlen)
    #targets = [torch.stack(captions),torch.stack(articles),torch.stack(titles),torch.stack(leads)]
    #targets = [captions,articles,titles,leads]
    #offsets = [captions_offset,articles_offset,titles_offset,leads_offset]

    #input_cap = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in ft_vocab.numericalize(caps)]
    #input_art = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in ft_vocab.numericalize(arts)]
    #input_tit = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in ft_vocab.numericalize(tits)]
    #input_lead = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in ft_vocab.numericalize(leds)]
    
    #input_cap = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in caps]
    #input_art = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in arts]
    #input_tit = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in tits]
    #input_lead = [(torch.LongTensor(i),torch.LongTensor(o)) for i,o in leds]
    
    """
    caps_i = torch.stack(caps_i)
    caps_o = torch.stack(caps_o)
    arts_i = torch.stack(arts_i)
    arts_o = torch.stack(arts_o)
    tits_i = torch.stack(tits_i)
    tits_o = torch.stack(tits_o)
    leds_i = torch.stack(leds_i)
    leds_o = torch.stack(leds_o)
    """
    """ # for testing
    input_cap = [(torch.LongTensor([0]),torch.LongTensor([0]))]*256
    input_art = input_cap
    input_tit = input_cap
    input_lead = input_cap
    """ 
    targets = [caps_i,caps_o,arts_i,arts_o,tits_i,tits_o,leds_i,leds_o]
    #targets = [caps_i,caps_o,arts_i,arts_o]



    #return images, subpics_list, targets, lengths, ids
    #return images, targets, lengths, ids
    return images, targets, ids


def get_loader_single(data_name, split, root, json, vocab, transform,
                      batch_size=100, shuffle=True,
                      num_workers=2, ids=None, collate_fn=collate_fn,
                      label='joint', random_miss=False,only_test=None,
                      text_encoder="default",train=True):

    if 'article' in data_name:
        if label in ['article','title','lead','caption']:
            dataset = ArticleDatasetUnimodal(root=root,
                              csv_path=json,
                              vocab_de=vocab[0],
                              vocab_fr=vocab[1],
                              transform=transform, 
                              ids=ids,label=label,
                              random_miss=random_miss,
                              only_test=only_test,
                              train=train)
            print ('[unimodal {} dataset created!]'.format(label))

        else:
            dataset = ArticleDataset(root=root,
                              csv_path=json,
                              vocab=vocab,
                              transform=transform, 
                              ids=ids,label=label,
                              random_miss=random_miss,
                              only_test=only_test)
            print ('[multimodal article dataset created!]')
    
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              #pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader, dataset


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              #pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    label = opt.label
    text_encoder = opt.text_encoder
    random_miss = opt.random_miss
    if ('article' in data_name):
        collate_fn_type = collate_fn
        if label in ['caption','article','lead','title']:
            collate_fn_type = collate_fn_unimodal
            print ('[using collate_fn_unimodal]')
            opt.only_test = False
        transform = get_transform(data_name, 'train',opt)
        train_loader,train_dataset = get_loader_single(data_name, 'train',
                                        opt.image_path,
                                        opt.caption_path,
                                        vocab, transform=transform, 
                                        batch_size=batch_size, 
                                        shuffle=True,
                                        num_workers=workers, 
                                        collate_fn=collate_fn_type,
                                        label=label,
                                        random_miss=random_miss,
                                        text_encoder=text_encoder,
                                        train=True)
        transform = get_transform(data_name, 'val',opt)
        val_loader,val_dataset = get_loader_single(opt.data_name, 'val',
                                       opt.val_image_path,
                                       opt.val_caption_path,
                                       vocab, transform=transform,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn_type,
                                       label=label,
                                       random_miss=False,
                                       text_encoder=text_encoder,
                                       train=False)
        val_loader_cap, val_loader_art, val_loader_title, val_loader_lead = None,None,None,None
        if opt.only_test == True:
            val_loader_cap,_ = get_loader_single(opt.data_name, 'val',
                                       opt.val_image_path,
                                       opt.val_caption_path,
                                       vocab, transform=transform,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn,
                                       label = label,
                                       random_miss = False,
                                       only_test = 'caption')
            val_loader_art,_ = get_loader_single(opt.data_name, 'val',
                                       opt.val_image_path,
                                       opt.val_caption_path,
                                       vocab, transform=transform,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn,
                                       label = label,
                                       random_miss = False,
                                       only_test = 'article')
            val_loader_title,_ = get_loader_single(opt.data_name, 'val',
                                       opt.val_image_path,
                                       opt.val_caption_path,
                                       vocab, transform=transform,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn,
                                       label = label,
                                       random_miss = False,
                                       only_test = 'title')
            val_loader_lead,_ = get_loader_single(opt.data_name, 'val',
                                       opt.val_image_path,
                                       opt.val_caption_path,
                                       vocab, transform=transform,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn,
                                       label = label,
                                       random_miss = False,
                                       only_test = 'lead')
            return train_loader,val_loader, val_dataset, (val_loader_cap,val_loader_art,val_loader_title,val_loader_lead)
        return train_loader, val_loader, val_dataset
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, 'train', opt)
        train_loader = get_loader_single(opt.data_name, 'train',
                                         roots['train']['img'],
                                         roots['train']['cap'],
                                         vocab, transform, ids=ids['train'],
                                         batch_size=batch_size, shuffle=True,
                                         num_workers=workers,
                                         collate_fn=collate_fn,
                                         label=label,
                                         random_miss=random_miss)

        transform = get_transform(data_name, 'val', opt)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, crop_size, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    if 'article' in data_name:
        collate_fn_type = collate_fn
        if label in ['caption','article','lead','title']:
            collate_fn_type = collate_fn_unimodal
        test_loader = get_loader_single(data_name, 'test',
                                       opt.val_img_path,
                                       opt.val_caption_path,
                                       vocab, transform=None ,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn_type,
                                       label=label
                                       )
    else:
        # Build Dataset Loader
        roots, ids = get_paths(dpath, data_name, opt.use_restval)

        transform = get_transform(data_name, split_name, opt)
        test_loader = get_loader_single(opt.data_name, split_name,
                                        roots[split_name]['img'],
                                        roots[split_name]['cap'],
                                        vocab, transform, ids=ids[split_name],
                                        batch_size=batch_size, shuffle=False,
                                        num_workers=workers,
                                        collate_fn=collate_fn,
                                        label=label)

    return test_loader


def compute_inputs_and_offsets(df, label, vocab):
    """
    return a dictionary of {img_name:(input,offset)} pairs.
    """
    r = {}
    seqlen = None
    if label == "caption": seqlen = 40
    elif label == "article": seqlen = 150
    elif label == "title": seqlen = 20
    elif label == "lead": seqlen = 70

    for i,row in df.iterrows():
        text_tok = row[label].split(" ") # being processed before
        img_name = row['img_name']
        text_tok_padded = _padder_(text_tok,seqlen)
        text_i,text_o = vocab.numericalize([text_tok_padded])[0]  
        text_i = torch.LongTensor(text_i)
        text_o = torch.LongTensor(text_o)
        r[img_name] = (text_i,text_o)
        if (i+1) % 10000 == 0:
            print ("[{}/{}] complete.".format(i+1,len(df)))
    return r


def main(opt):
    """
    precompute subword indices & offsets, save in pkl file.
    """
    df = pd.read_csv(open(opt.caption_path,'rb'))
    ft_vocab = pickle.load(open(opt.ft_vocab_path,"rb"))

    print ("[computing captions...]")
    r_cap = compute_inputs_and_offsets(df,"caption",ft_vocab)
    with open("precomputed_caption_ft_subword_val.pkl", 'wb') as f:
        pickle.dump(r_cap, f)
    print ("[computing articles...]")
    r_art = compute_inputs_and_offsets(df,"article",ft_vocab)
    with open("precomputed_article_ft_subword_val.pkl", 'wb') as f:
        pickle.dump(r_art, f)
    print ("[computing titles...]")
    r_tit = compute_inputs_and_offsets(df,"title",ft_vocab)
    with open("precomputed_title_ft_subword_val.pkl", 'wb') as f:
        pickle.dump(r_tit, f)
    print ("[computing leads...]")
    r_led = compute_inputs_and_offsets(df,"lead",ft_vocab)
    with open("precomputed_lead_ft_subword_val.pkl", 'wb') as f:
        pickle.dump(r_led, f)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', default='./',
            help='path to caption csv file.')
    parser.add_argument('--ft_vocab_path', default='./',
            help='path to fastText vocab file.')
    opt = parser.parse_args()
    main(opt)



