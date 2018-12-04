import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm
import pickle
import numpy as np
import gensim
from collections import OrderedDict

# torch-goodies
from embeddings import FastTextVocab
from embeddings import FastTextEmbeddings

# flair model
from sequence_tagger_model import *
#import flair embeddings

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = True

from utils import *
# models
from attention import Attention

from transformer.Models import Encoder
from transformer.SubLayers import MultiHeadAttention, _PositionwiseFeedForward
from transformer.Layers import EncoderLayer
# RNNs
"""
from fastai.learner import *
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.text import *
from fastai.lm_rnn import *
"""
# CNNs
from textcnn import CNN_Text
from tcn import TemporalConvNet
from gated_cnn import GatedCNN 
from VDCNN import VDCNN


def load_my_vocab_with_pretrained_vec(pretrained_model_path,vocab,d_word_vec=300):
    pretrained_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model_path)
    embed_matrix = np.ramdon.uniform(-0.1,0.1,(len(vocab),d_word_vec))
    count1,count2=0,0
    for i,(key,val) in enumerate(vocab.word2idx.items()):
        try:
            emb_matrix [i,:] = pretrained_model[key]
            count1 += 1
        except:
            # key not in pretrained model
            count2 += 1
    print ("[ {} pre-trained vec used. ]".format(count1))
    print ("[ {} not using pre-trained vec. ]".format(count2))
    return embed_matrix

class DotDict(dict):
    def __getattr__(self, name):
        return self[name]

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()


        # define a cnn for text
        self.text_cnn = models.squeezenet1_0()
        for param in self.text_cnn.parameters():
            param.requires_grad = True

        self.init_weights()

    def get_cnn(self, arch, pretrained, 
            model_path="/mnt/storage01/fangyu/places365_pretrained_models/resnet50_places365.pth.tar"):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            if 'places365' in arch:
                print("=> using pre-trained model '{}'".format(arch))
                model = models.__dict__['resnet50'](num_classes=365)
                checkpoint = torch.load(model_path,map_location=lambda storage, loc:storage)
                state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
                model.load_state_dict(state_dict)
                print (arch,'loaded.')
            else:
                print("=> using pre-trained model '{}'".format(arch))
                model = models.__dict__[arch](pretrained=True)

        else:
            print("=> creating model '{}<<<<<<'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)




class EncoderTextTransformer(nn.Module):

    def __init__(self, vocab, word_dim, embed_size, num_layers,
                 use_abs=False, pretrained_emb=False,freeze_emb=False, test=False, 
                 resume=False,both=False,embeddings=None,label="caption",pretrained_embeddingbag_path="",lang='de',attn_out=False):
        super(EncoderTextTransformer, self).__init__()
        self.vocab_de = vocab[0]
        self.vocab_fr = vocab[1]
        self.word_dim = word_dim
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.pretrained_emb = pretrained_emb
        self.freeze_emb = freeze_emb
        self.test = test
        self.resume = resume
        self.both = both
        self.pretrained_embeddingbag_path = pretrained_embeddingbag_path # to be fixed
        self.word_dim = 300
        self.attn_out = attn_out
        #self.attn_out = True

        if test: 
            print ('testing mode')

        # word embedding
        print ('[init EmbeddingBags in EncoderText...]')
        #lang = 'de'
        #path_to_binary_file='/mnt/storage01/fangyu/fasttext_embeddings/cc.'+lang+'.300.bin'
        #embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
        #self.vocab = FastTextVocab(
        #        include_word=True,
        #        min_subword=4, max_subword=6, # seems cc only works with 5 5
        #        word_min_freq=1, subword_min_freq=1)
        #self.vocab.create_index_mapping(
        #        embeddings, vocab=self.vanilla_vocab.word2idx )
        #print ('declare  embedding...')
        print ('[declare models...]')
        self.embed_size = 1024
        seq_max_len = None
        if label == "article": seq_max_len = 150
        elif label == "caption": seq_max_len = 40
        elif label == "title": seq_max_len = 20
        elif label == "lead": seq_max_len = 70
        print ("[label: {}, seq_max_len: {}]".format(label,seq_max_len))
        self.transformer = Encoder(len(self.vocab_de),len(self.vocab_fr), seq_max_len,d_word_vec=300,
                n_layers=1,n_head=6,d_k=64,d_v=64,d_model=300,d_inner=1024,dropout=0.01) # 0.01
        embedding_sum_de = nn.EmbeddingBag(len(self.vocab_de), 300, mode='sum')
        embedding_sum_fr = nn.EmbeddingBag(len(self.vocab_fr), 300, mode='sum')
        
        embedding_sum_de.load_state_dict(torch.load('./vocab_de_4_6_1_3_embedding_bag_.pth'))
        embedding_sum_fr.load_state_dict(torch.load('./vocab_fr_4_6_1_3_embedding_bag_.pth'))
        #embeddings.init_weights(embedding_sum.weight.data, self.vocab.vocab())
        #torch.save(embedding_sum.state_dict(),'./vocab_'+lang+'_4_6_1_3_embedding_bag_.pth')
        #print ('embedding bag saved.')

        self.transformer.src_word_emb_de = embedding_sum_de
        self.transformer.src_word_emb_fr = embedding_sum_fr
        #self.transformer.src_word_emb = nn.Embedding(self.vocab_size, 300)
        #self.transformer.src_word_emb.weight.data.uniform_(-0.1,0.1)
        
        #self.src_word_emb = self.embedding_sum

        #self.gru = nn.GRU(300,1024,1, batch_first=True)

        #self.pool = nn.AdaptiveMaxPool1d(4)
        self._pool = nn.AdaptiveMaxPool1d(1)
        #self.fc = nn.Linear(4*300,1024)
        
        #""" # commneted to test no translator situation
        #if lang == 'fr':
        self.transformer.translator = nn.Linear(300,300)
        #"""
        #    self.transformer.weight = torch.tensor(load_matrix('/home/fangyu/repos/fastText/alignment/res/subword.fr-de.vec-mat'),dtype=torch.float).cuda().t() # transpose cuz originaly np.dot(input,m), now F.linear is xA^T


        if self.freeze_emb:
            self.transformer.src_word_emb_de.requires_grad = False
            self.transformer.src_word_emb_fr.requires_grad = False
            print ("[Transformer fr word embedding freezed!]")
        

        #self.init_weights()

    def init_weights(self):
        """
        if (not self.pretrained_emb) or (self.test) or (self.resume):
            print ('randomly init embedding weights...')
            self.embedding_sum.weight.data.uniform_(-0.1,0.1)
            #self.embedding_sum.init_weights(self.embeddings.embedding_weights(), self.vocab.vocab())
            #self.embed.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_cap.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_art.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_title.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_lead.weight.data.uniform_(-0.1, 0.1) # original init
        else:
            self.embedding_sum.load_state_dict(torch.load('./embedding_bag.pth'))
            print (self.embedding_sum)
        """
                                       

    def forward(self, x):
        #"""Handles variable size captions
        #"""
        #src_seq_list = [self.src_word_emb(x[0][i],x[1][i]) for i in range(len(x[0]))]
        #src_seq = torch.stack(src_seq_list)
        #out,_ = self.gru(src_seq)
        #out,attn = self.transformer(x,return_attns=True)
        out,attn = self.transformer(x,return_attns=True)
        if self.attn_out == True:
            attn = attn[0]
            fname = 'scores.txt'
            fout = io.open(fname, 'a', encoding='utf-8')
            for i in range(out.size()[0]):
                m = list(attn[i:i+6].mean(0).mean(0))
                fout.write(" ".join(map(lambda a: "%.4f" % a, m)) + "\n")
                # write to pickle
            fout.close()

        out = self._pool(out.transpose(1,2)).view(len(out),-1) # use pooling rather than nn.gather
        
        #out = self.fc(out) # onlu used in multi-layer models

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
            
        # return out,attn
        return out



class EncoderTextMultimodalTransformer(nn.Module):

    def __init__(self, vocab, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, pretrained_emb=False,freeze_emb=False, test=False, 
                 resume=False,both=False,embeddings=None,label="caption",pretrained_embeddingbag_path=""):
        super(EncoderTextMultimodalTransformer, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.pretrained_emb = pretrained_emb
        self.freeze_emb = freeze_emb
        self.test = test
        self.resume = resume
        self.both = both
        self.pretrained_embeddingbag_path = pretrained_embeddingbag_path # to be fixed
        self.word_dim = 300

        if test: 
            print ('testing mode')

        # word embedding
        #print ('[init EmbeddingBags in EncoderText...]')
        #path_to_binary_file='/mnt/storage01/fangyu/fasttext_embeddings/wiki.de.bin'
        #embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
        #fine_tuning = True
        #subword_only = False
        #min_subword = 3
        #max_subword = 6
        #word_min_freq = 1
        #subword_min_freq = 1
        #self.vocab = FastTextVocab(embeddings,
        #        vocab=self.vocab.word2idx if fine_tuning else None,
        #        include_word=not subword_only,
        #        min_subword=min_subword, max_subword=max_subword,
        #        word_min_freq=word_min_freq, subword_min_freq=subword_min_freq)
 
        #print ('declare  embedding...')
        print ('[declare models...]')
        self.embed_size = 1024
        self.transformer_cap = Encoder(len(self.vocab),70,d_word_vec=300,
                n_layers=1,n_head=6,d_k=64,d_v=64,d_model=300,d_inner=1024,dropout=0.01)
        self.transformer_art = Encoder(len(self.vocab),150,d_word_vec=300,
                n_layers=1,n_head=6,d_k=64,d_v=64,d_model=300,d_inner=1024,dropout=0.01)
        self.transformer_tit = Encoder(len(self.vocab),20,d_word_vec=300,
                n_layers=1,n_head=6,d_k=64,d_v=64,d_model=300,d_inner=1024,dropout=0.01)
        self.transformer_led = Encoder(len(self.vocab),70,d_word_vec=300,
                n_layers=1,n_head=6,d_k=64,d_v=64,d_model=300,d_inner=1024,dropout=0.01)
        self.embedding_sum = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        self.embedding_sum.load_state_dict(torch.load('./vocab_4_6_1_3_embedding_bag.pth'))

        #"""
        # multimodal models share one embed
        self.transformer_cap.src_word_emb = self.embedding_sum
        self.transformer_art.src_word_emb = self.embedding_sum
        self.transformer_tit.src_word_emb = self.embedding_sum
        self.transformer_led.src_word_emb = self.embedding_sum
        #"""
        #"""
        # separate embeddings
        #self.transformer_cap.src_word_emb = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        #self.transformer_cap.src_word_emb.load_state_dict(torch.load('./vocab_4_6_1_3_embedding_bag.pth'))
        #self.transformer_art.src_word_emb = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        #self.transformer_art.src_word_emb.load_state_dict(torch.load('./vocab_4_6_1_3_embedding_bag.pth'))
        #self.transformer_tit.src_word_emb = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        #self.transformer_tit.src_word_emb.load_state_dict(torch.load('./vocab_4_6_1_3_embedding_bag.pth'))
        #self.transformer_led.src_word_emb = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        #self.transformer_led.src_word_emb.load_state_dict(torch.load('./vocab_4_6_1_3_embedding_bag.pth'))
        #"""

        #"""
        _state_dict_art = torch.load("/home/fangyu/data/models_for_demo/article_transformer_1layer_FOR_VUE_150seqlen/model_best.pth.tar")["model"][1]
        state_dict_art = {}
        for k,v in _state_dict_art.items():
            if k.startswith("transformer"):
                state_dict_art[k[12:]] = v
        self.transformer_art.load_state_dict(state_dict_art)
        print ("art model loaded")

        _state_dict_tit = torch.load("/home/fangyu/data/models_for_demo/title_transformer_1layer_6heads_64k_v_0.01dropout/model_best.pth.tar")["model"][1]
        state_dict_tit = {}
        for k,v in _state_dict_tit.items():
            if k.startswith("transformer"):
                state_dict_tit[k[12:]] = v
        self.transformer_tit.load_state_dict(state_dict_tit)
        print ("tit model loaded")

        _state_dict_led = torch.load("/home/fangyu/data/models_for_demo/lead_transformer_1layer_6heads_64k_v_0.01dropout/model_best.pth.tar")["model"][1]
        state_dict_led = {}
        for k,v in _state_dict_led.items():
            if k.startswith("transformer"):
                state_dict_led[k[12:]] = v
        self.transformer_led.load_state_dict(state_dict_led)
        print ("led model loaded")
        
        _state_dict_cap = torch.load("/home/fangyu/data/models_for_demo/caption_transformer_1layer_6heads_64k_v_0.01dropout/model_best.pth.tar")["model"][1]
        state_dict_cap = {}
        for k,v in _state_dict_cap.items():
            if k.startswith("transformer"):
                state_dict_cap[k[12:]] = v
        self.transformer_cap.load_state_dict(state_dict_cap)
        print ("cap model loaded")
        #"""


        #embeddings.init_weights(self.embedding_sum.weight.data, self.vocab.vocab())
        #torch.save(self.embedding_sum.state_dict(),'./vocab_article_4_6_1_3_embedding_bag.pth')
        #print ('embedding bag saved.')

        self._pool = nn.AdaptiveMaxPool1d(1)
        self.fuse_pool = nn.AdaptiveMaxPool1d(1)

        #self.self_attn = Attention(1024)
        
        
        self.transformer_fuse = EncoderLayer(1024,1024,1,8,8,dropout=0.01)
        self.fc = nn.Sequential( 
                nn.Linear(4096,4096),
                nn.ReLU(),
                nn.Linear(4096,1024))

        

        #self.init_weights()

    def init_weights(self):
        if (not self.pretrained_emb) or (self.test) or (self.resume):
            print ('randomly init embedding weights...')
            self.embedding_sum.weight.data.uniform_(-0.1,0.1)
            #self.embedding_sum.init_weights(self.embeddings.embedding_weights(), self.vocab.vocab())
            #self.embed.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_cap.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_art.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_title.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_lead.weight.data.uniform_(-0.1, 0.1) # original init
        else:
            self.embedding_sum.load_state_dict(torch.load('./embedding_bag.pth'))
            print (self.embedding_sum)
                                       

    def forward(self, x):
        """Handles variable size captions
        """
        bs = len(x[0][0])
        out_cap = self.transformer_cap(x[0])[0]
        out_art = self.transformer_art(x[1])[0]
        out_tit = self.transformer_tit(x[2])[0]
        out_led = self.transformer_led(x[3])[0]
        #print ("cap {}; art {}; tit {}; led {}".format(out_cap.size(),out_art.size(),out_tit.size(),out_led.size()))
        out_cap = self._pool(out_cap.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        out_art = self._pool(out_art.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        out_tit = self._pool(out_tit.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        out_led = self._pool(out_led.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        #print ("cap {}; art {}; tit {}; led {}".format(out_cap.size(),out_art.size(),out_tit.size(),out_led.size()))
        
        #out_fused = out_cap + out_art + out_tit + out_led
        
        #out_fused = torch.cat([out_cap,out_art,out_tit,out_led],1)
        
        # direct max pool
        #out_fused = torch.stack([out_cap,out_art,out_tit,out_led]).transpose(0,1)
        #out_fused = self.fuse_pool(out_fused.transpose(1,2)).squeeze(2)

        #""" Attn + neural fuse
        out_fused = torch.stack([out_cap,out_art,out_tit,out_led]).transpose(0,1)

        #out_fused = torch.stack([out_cap,out_art]).transpose(0,1)

        #out_fused = out_fused.view(len(out_cap),-1)
        #print (out_fused.size())
        
        #out_fused,_ = self.self_attn(out_fused)
        # transformer fuse
        out_fused, fuse_attn = self.transformer_fuse(out_fused)
        
        #out_fused = self.fuse_pool(out_fused.transpose(1,2)).squeeze(2)
        out_fused = out_fused.contiguous().view(bs,-1)
        out_fused = self.fc(out_fused)
        #"""

        out = out_fused
        
        #out = self.fc(out) # onlu used in multi-layer models

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
    
        return out



class EncoderTextUnimodal(nn.Module):

    def __init__(self, vocab, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, pretrained_emb=False,freeze_emb=False, test=False, 
                 resume=False,both=False,embeddings=None,label="caption",pretrained_embeddingbag_path=""):
        super(EncoderTextUnimodal, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.pretrained_emb = pretrained_emb
        self.freeze_emb = freeze_emb
        self.test = test
        self.resume = resume
        self.both = both
        self.pretrained_embeddingbag_path = pretrained_embeddingbag_path # to be fixed
        #self.loaded_weights = torch.load('/home/fangyu/data/DE_model_dropout_0.1_1cycle_10epochs.pt', map_location=lambda storage, loc: storage)
        #self.ulm_embed_weights = to_np(self.loaded_weights['0.encoder.weight'])
        #self.vocab_size = self.ulm_embed_weights.shape[0] # 60002
        #self.word_dim = self.ulm_embed_weights.shape[1] # 400
        self.word_dim = 300

        if test: 
            print ('testing mode')

        # word embedding
        #print ('[init EmbeddingBags in EncoderText...]')
        #path_to_binary_file='/mnt/storage01/fangyu/fasttext_embeddings/wiki.de.bin'
        #embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
        #fine_tuning = True
        #subword_only = False
        #min_subword = 3
        #max_subword = 6
        #word_min_freq = 1
        #subword_min_freq = 1
        #self.vocab = FastTextVocab(embeddings,
        #        vocab=self.vocab.word2idx if fine_tuning else None,
        #        include_word=not subword_only,
        #        min_subword=min_subword, max_subword=max_subword,
        #        word_min_freq=word_min_freq, subword_min_freq=subword_min_freq)
 
        #print ('declare  embedding...')
        self.embedding_sum = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        self.embedding_sum.load_state_dict(torch.load('./vocab_article_4_6_1_3_embedding_bag.pth'))
        #embeddings.init_weights(self.embedding_sum.weight.data, self.vocab.vocab())
        #torch.save(self.embedding_sum.state_dict(),'./vocab_article_4_6_1_3_embedding_bag.pth')
        #print ('embedding bag saved.')

        #self.embed =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_cap =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_art =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_title =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_lead =  nn.Embedding(self.vocab_size, self.word_dim)
        
        print ('[declare models...]')


        # multi-head attention
        self.attn = MultiHeadAttention(n_head=4,d_model=300,d_k=64,d_v=64,dropout=0.05)

        # primitive RNNs
        self.embed_size = 1024
        self.rnn = None
        if label == "article":
            self.rnn = nn.LSTM(self.word_dim, int(self.embed_size/2), num_layers, batch_first=True,bidirectional=True)
        else:
            self.rnn = nn.GRU(self.word_dim, self.embed_size, num_layers, batch_first=True)
        print (self.rnn) 

        '''# load prev weights
        cap_model_path = '/mnt/storage01/fangyu/sota_models/caption_models/caption_1layer_GRU_pretrained/model_best.pth.tar'
        art_model_path = '/mnt/storage01/fangyu/sota_models/article_models/article_1layer_LSTM_pretrained/model_best.pth.tar'
        title_model_path = '/mnt/storage01/fangyu/sota_models/title_models/title_1layer_GRU_pretrained/model_best.pth.tar'
        lead_model_path = '/mnt/storage01/fangyu/sota_models/lead_models/lead_1layer_GRU_pretrained/model_best.pth.tar'

        # load cap  model
        checkpoint = torch.load(cap_model_path)
        state_dict = checkpoint['model']
        #for k,v in state_dict[1].items(): # state_dict[0] is CNN
        #    print (k,v.shape)
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_cap.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_cap.load_state_dict(rnn_weight)
        #print ('caption model loaded.')

        # load title model
        checkpoint = torch.load(title_model_path)
        state_dict = checkpoint['model']
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_title.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_title.load_state_dict(rnn_weight)
        #print ('title model loaded.')
        
        # load lead model
        checkpoint = torch.load(lead_model_path)
        state_dict = checkpoint['model']
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_lead.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_title.load_state_dict(rnn_weight)
        #print ('lead model loaded.')

        #self.embed_lead.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_lead.load_state_dict(rnn_weight)
        
        #checkpoint = torch.load(art_model_path)
        #state_dict = checkpoint['model']
        #for k,v in state_dict[1].items():
        #    print (k,v.shape)
        #rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        #self.embed_art.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_art.load_state_dict(rnn_weight)
        ''' # load pretrained 1 layer RNN weights

        #self.rnn = nn.GRU(self.word_dim, embed_size, num_layers, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        

        # some filters to reduce embed size
        #self.cnn_reduce_size = cnn_reduce_size(args)
        
        '''# load pretrained RNN weights to model
        # copy pretrained weight
        
        #print (self.ulm_encoder) # print structure
        inited_weights = self.ulm_encoder.state_dict()
        #for param in inited_weights:
        #    print (param.data.shape)
        #for k,v in self.loaded_weights.items():
            #print (param,':',self.loaded_weights[param].shape)
            #print (k,v.shape)
        
        
        encoder_weights = {k[2:]: v for k,v in self.loaded_weights.items() if k.startswith('0.')}
        #print (self.loaded_weights.items())
            
        for k,v in encoder_weights.items():
            # skip embed layers, assign in "init_weights()"
            if k.startswith('encoder'):
                continue
            # only use pretrained in rnn layer 0
            #if k.startswith('rnns.1'): 
            #    break
            #print (param,':',self.loaded_weights[param].shape)
            #inited_weights[k] = encoder_weights[k]
            inited_weights[k].copy_(encoder_weights[k])
            print (k,v.shape, 'loaded')


        # copy weights to our model
        self.ulm_encoder.load_state_dict(inited_weights)
       '''

        # ensure grads
        #for param in self.ulm_encoder.parameters():
        #    param.requires_grad = True
        '''
        for i,(name,param) in enumerate(self.ulm_encoder.named_parameters()):
            if i == 0 or i >=9:
                param.requires_grad = True
                print (name,'not freeze')
            else:
                param.requires_grad = False
                print (name,'freeze')

        #sys.exit(0)
        '''
        #self.ulm_fc_1 = nn.Sequential(
        #        nn.Linear(30,20),
        #        nn.MaxPool1d(2), 
        #        )
        #seqlen = 32
        #self.ulm_fc_2 = nn.Sequential(
        #       nn.Linear(seqlen*10,1024),
               #nn.BatchNorm1d(1024)
        #        )
        

        #self.init_weights()

    def init_weights(self):
        if (not self.pretrained_emb) or (self.test) or (self.resume):
            print ('randomly init embedding weights...')
            self.embedding_sum.weight.data.uniform_(-0.1,0.1)
            #self.embedding_sum.init_weights(self.embeddings.embedding_weights(), self.vocab.vocab())
            #self.embed.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_cap.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_art.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_title.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_lead.weight.data.uniform_(-0.1, 0.1) # original init
        else:
            self.embedding_sum.load_state_dict(torch.load('./embedding_bag.pth'))
            print (self.embedding_sum)
                                       

    def forward(self, x):
        """Handles variable size captions
        """
        l = len(x[0])
        #print (len(x[0]),len(x[1]),len(x[2]),len(x[3]),len(x[4]),len(x[5]),len(x[6]),len(x[7]))
        text =[self.embedding_sum(x[0][i],x[1][i]) for i in range(l)]
        text = torch.stack(text)

        text = self.attn(text,text,text)[0]

        out_text, _ = self.rnn(text)
        
        out = self.pool(out_text.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
    
        return out

class EncoderText(nn.Module):

    def __init__(self, vocab, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False, pretrained_emb=False,freeze_emb=False, test=False, 
                 resume=False,both=False,embeddings=None):
        super(EncoderText, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.pretrained_emb = pretrained_emb
        self.freeze_emb = freeze_emb
        self.test = test
        self.resume = resume
        self.both = both
        #self.loaded_weights = torch.load('/home/fangyu/data/DE_model_dropout_0.1_1cycle_10epochs.pt', map_location=lambda storage, loc: storage)
        #self.ulm_embed_weights = to_np(self.loaded_weights['0.encoder.weight'])
        #self.vocab_size = self.ulm_embed_weights.shape[0] # 60002
        #self.word_dim = self.ulm_embed_weights.shape[1] # 400
        self.word_dim = 300

        if test: 
            print ('testing mode')

        # word embedding
        #print ('[init EmbeddingBags in EncoderText...]')
        #path_to_binary_file='/mnt/storage01/fangyu/fasttext_embeddings/wiki.de.bin'
        #embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
        #fine_tuning = True
        #subword_only = False
        #min_subword = 3
        #max_subword = 6
        #word_min_freq = 1
        #subword_min_freq = 1
        #self.vocab = FastTextVocab(embeddings,
        #        vocab=self.vocab.word2idx if fine_tuning else None,
        #        include_word=not subword_only,
        #        min_subword=min_subword, max_subword=max_subword,
        #        word_min_freq=word_min_freq, subword_min_freq=subword_min_freq)
 
        #print ('declare  embedding...')
        self.embedding_sum = nn.EmbeddingBag(len(self.vocab), 300, mode='sum')
        #embeddings.init_weights(self.embedding_sum.weight.data, self.vocab.vocab())
        #torch.save(self.embedding_sum.state_dict(),'./embedding_bag.pth')
        #print ('embedding bag saved.')
        self.embedding_sum.load_state_dict(torch.load('./embedding_bag.pth'))
        #print (self.embedding_sum)
        #self.embed =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_cap =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_art =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_title =  nn.Embedding(self.vocab_size, self.word_dim)
        #self.embed_lead =  nn.Embedding(self.vocab_size, self.word_dim)
        
        print ('[declare models...]')
        args = DotDict()
        args.embed_num = vocab_size
        args.embed_dim = word_dim
        args.class_num = 1024
        args.kernel_num = 256
        args.kernel_sizes = [1,3,5,10,20]
        args.dropout = 0.5
        #self.cnn_enc = CNN_Text(args)
        #self.cnn_enc = TemporalConvNet(word_dim,num_channels=[256,256,512,512,1024,1024],kernel_size=2,dropout=args.dropout)
        # caption embedding
        # step 1: a CNN filter

        # primitive RNNs
        self.embed_size = 1024
        self.rnn_cap = nn.GRU(self.word_dim, self.embed_size, num_layers, batch_first=True)
        self.rnn_art = nn.BiLSTM(self.word_dim, self.embed_size/2, num_layers, batch_first=True)
        self.rnn_title = nn.GRU(self.word_dim, self.embed_size, num_layers, batch_first=True)
        self.rnn_lead = nn.GRU(self.word_dim, self.embed_size, num_layers, batch_first=True)
        print (self.rnn_cap) 
        print (self.rnn_art) 
        print (self.rnn_title) 
        print (self.rnn_lead) 

        '''# load prev weights
        cap_model_path = '/mnt/storage01/fangyu/sota_models/caption_models/caption_1layer_GRU_pretrained/model_best.pth.tar'
        art_model_path = '/mnt/storage01/fangyu/sota_models/article_models/article_1layer_LSTM_pretrained/model_best.pth.tar'
        title_model_path = '/mnt/storage01/fangyu/sota_models/title_models/title_1layer_GRU_pretrained/model_best.pth.tar'
        lead_model_path = '/mnt/storage01/fangyu/sota_models/lead_models/lead_1layer_GRU_pretrained/model_best.pth.tar'

        # load cap  model
        checkpoint = torch.load(cap_model_path)
        state_dict = checkpoint['model']
        #for k,v in state_dict[1].items(): # state_dict[0] is CNN
        #    print (k,v.shape)
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_cap.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_cap.load_state_dict(rnn_weight)
        #print ('caption model loaded.')

        # load title model
        checkpoint = torch.load(title_model_path)
        state_dict = checkpoint['model']
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_title.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_title.load_state_dict(rnn_weight)
        #print ('title model loaded.')
        
        # load lead model
        checkpoint = torch.load(lead_model_path)
        state_dict = checkpoint['model']
        rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        self.embed_lead.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_title.load_state_dict(rnn_weight)
        #print ('lead model loaded.')

        #self.embed_lead.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_lead.load_state_dict(rnn_weight)
        
        #checkpoint = torch.load(art_model_path)
        #state_dict = checkpoint['model']
        #for k,v in state_dict[1].items():
        #    print (k,v.shape)
        #rnn_weight = {k[4:]:v for k,v in state_dict[1].items() if k.startswith('rnn')}
        #self.embed_art.weight.data.copy_(state_dict[1]['embed.weight'])
        #self.rnn_art.load_state_dict(rnn_weight)
        ''' # load pretrained 1 layer RNN weights

        #self.rnn = nn.GRU(self.word_dim, embed_size, num_layers, batch_first=True)
        self.pool_cap = nn.AdaptiveMaxPool1d(1)
        self.pool_art = nn.AdaptiveMaxPool1d(1)
        self.pool_title = nn.AdaptiveMaxPool1d(1)
        self.pool_lead = nn.AdaptiveMaxPool1d(1)
        
        self.fuse_pool = nn.AdaptiveMaxPool1d(1)
        self.non_linear_fc = nn.Sequential(
                nn.Linear(1024*4,1024*4),
                nn.ReLU(),
                nn.Linear(1024*4,1024),
                nn.ReLU()
                )
        #self.attention = Attention(self.embed_size)
        #print (self.attention)

        # some filters to reduce embed size
        #self.cnn_reduce_size = cnn_reduce_size(args)


        self.init_weights()

    def init_weights(self):
        if (not self.pretrained_emb) or (self.test) or (self.resume):
            print ('randomly init embedding weights...')
            #self.embedding_sum.init_weights(self.embeddings.embedding_weights(), self.vocab.vocab())
            #self.embed.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_cap.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_art.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_title.weight.data.uniform_(-0.1, 0.1) # original init
            #self.embed_lead.weight.data.uniform_(-0.1, 0.1) # original init
        else:
            #print ('loading pretrained embedding weights...')
            model = gensim.models.KeyedVectors.load_word2vec_format('/home/fangyu/downloads/wiki.de.vec')
            #model = gensim.models.KeyedVectors.load_word2vec_format('/data/retina_data/wiki.de.vec')
                #'/mnt/storage01/fangyu/fasttext_embeddings/wiki.de.vec' )
            #print ('pretrained wordvec loaded')
                        
            '''
            # for previous FastText pre-trained embed
            count1,count2 = 0,0
            emb_matrix = np.random.uniform(-0.1,0.1,(self.vocab_size,self.word_dim))
            print ('emb_matrix:',emb_matrix.shape)
            #emb_matrix = np.zeros((self.vocab_size, self.word_dim),dtype=np.float32) # use original (loaded vocab) vocab size
            for i,(key,value) in enumerate(self.vocab.word2idx.items()):
                #rint (key)
                try:
                    emb_matrix[i,:] = model[key]
                    #print (key,'replaced')
                    count1 += 1
                except:
                    #print (key,'not in pretrained')
                    count2 += 1 
            #self.embed.weight.data.copy_( torch.from_numpy(emb_matrix))
            #self.ulm_encoder.encoder.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.ulm_encoder.encoder_with_dropout.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
            #print (emb_matrix.shape)
            #print (self.VDCNN.embedding.weight.data.size())
            #self.VDCNN.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

            print (count1, 'pre-trained vec used')
            print (count2, 'words in vocab not using pre-trained weights')
            
            np.save('fasttext_word_embed.npy',emb_matrix)
            print ('fasttext word embbeding saved.')
            #self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
            '''
            # load from numpy
            emb_matrix = np.load('/home/fangyu/data/fasttext_word_embed.npy')

            #self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
            #if self.freeze_emb:
            #    self.embed.weight.requires_grad = False

            #self.embed = nn.Embedding(self.vocab_size, word_dim)
            #self.embed.weight.data.copy_(torch.from_numpy(new_w))
            #prev_states = self.ulm_encoder.state_dict()
            #prev_states['encoder.weight'].copy_(torch.from_numpy(new_w))
            #prev_states['encoder_with_dropout.embed.weight'].copy_(torch.from_numpy(new_w))
            #self.ulm_encoder.load_state_dict(prev_states)
            
            self.embed.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.embed_cap.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.embed_art.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.embed_title.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.embed_lead.weight.data.copy_(torch.from_numpy(emb_matrix))

            #self.ulm_encoder.encoder.weight.data.copy_(torch.from_numpy(emb_matrix))
            #self.ulm_encoder.encoder_with_dropout.embed.weight.data.copy_(torch.from_numpy(emb_matrix))

            #self.gated_cnn.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

            '''# Gated CNN
            #self.gated_cnn = GatedCNN(32,100000,self.embed_size,10,(5,self.embed_size),64,5,1024)
            self.gated_cnn = torch.load('/home/fangyu/data/gated_cnn_de_wiki_LM_best-32seqlen_55.55.pt')
            #prev_weights = torch.load('/home/fangyu/data/gated_cnn_de_wiki_LM_best-32seqlen_55.55.pt').parameters()
            # substitute last layer
            self.gated_cnn.fc = nn.Linear(2048,1024)
            print (self.gated_cnn)
            itos_gcnn = pickle.load(open('./vocab/wiki_100000_vocab.pkl','rb'))
            stoi_gcnn = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos_gcnn)})
            #print (stoi_ulm)
            #sys.exit(0)
            
            # load ulm pretrained embed
            for name,param in self.gated_cnn.embedding.named_parameters():
                print (name,param.size())
                embed_weights = param.data # loaded pre-trained embedding
            row_m = embed_weights.mean(0) # to be assigned to embed's rows
            #vs = embed_weights.shape[0] # 60002
            word_dim = embed_weights.shape[1]
            #new_w = np.zeros((vs, word_dim),dtype=np.float32)
            new_w = np.zeros((self.vocab_size, word_dim),dtype=np.float32) # use original (loaded vocab) vocab size
            print ('original vocab_size:',len(self.vocab.word2idx))
            print (new_w.shape)
            count1, count2 = 0,0
            for i,(key,val) in enumerate(self.vocab.word2idx.items()):
                #print (i,key,val)
                r = stoi_gcnn[key]
                if r>= 0:
                    count1 += 1
                    new_w[i] = embed_weights[r]
                else:
                    count2 += 1
                    new_w[i] = row_m
                #new_w[i] = embed_weights[r] if r>=0 else row_m
            print ('new word embedding size:',new_w.shape)
            print (count1, 'pre-trained vec used')
            print (count2, 'words in vocab not using pre-trained weights')
            '''
            

    def forward(self, x):
        """Handles variable size captions
        """
        l = len(x[0])
        #print (len(x[0]),len(x[1]),len(x[2]),len(x[3]),len(x[4]),len(x[5]),len(x[6]),len(x[7]))
        caps=[self.embedding_sum(x[0][i],x[1][i]) for i in range(l)]
        arts=[self.embedding_sum(x[2][i],x[3][i]) for i in range(l)]
        tits=[self.embedding_sum(x[4][i],x[5][i]) for i in range(l)]
        leds=[self.embedding_sum(x[6][i],x[7][i]) for i in range(l)]
        #arts=[self.embedding_sum(i_o[0],i_o[1]) for i_o in x[1]]
        #tits=[self.embedding_sum(i_o[0],i_o[1]) for i_o in x[2]]
        #leds=[self.embedding_sum(i_o[0],i_o[1]) for i_o in x[3]]
        #caps = [ self.embedding_sum(i,o) for i,o in x[0]]
            #print (tmp.size())
        caps = torch.stack(caps)
        arts = torch.stack(arts)
        tits = torch.stack(tits)
        leds = torch.stack(leds)
        #sys.exit(0)
        #x_cap = self.embedding_sum(x[0])
        #x_art = self.embedding_sum(x[1])
        #x_title = self.embedding_sum(x[2])
        #x_lead = self.embedding_sum(x[3])
        #print(x_cap.size())

        #x_cap = self.embed(x[0])
        #x_art = self.embed(x[1])
        #x_title = self.embed(x[2])
        #x_lead = self.embed(x[3])
        x_cap = caps
        x_art = arts
        x_title = tits
        x_lead = leds

        out_cap, _ = self.rnn_cap(x_cap)
        out_art, _ = self.rnn_art(x_art)
        out_title, _ = self.rnn_title(x_title)
        out_lead, _ = self.rnn_lead(x_lead)
        
        pooled_cap = self.pool_cap(out_cap.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        pooled_art = self.pool_art(out_art.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        pooled_title = self.pool_art(out_title.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        pooled_lead = self.pool_art(out_lead.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather

        #concated_feature = torch.stack([pooled_cap,pooled_art,pooled_title,pooled_lead]).transpose(0,1).transpose(1,2)
        concated_feature = torch.cat((pooled_cap,pooled_art,pooled_title,pooled_lead),1)
        #concated_feature = torch.stack([pooled_cap,pooled_art]).transpose(0,1).transpose(1,2)
        out = self.non_linear_fc(concated_feature)
        #out = self.fuse_pool(concated_feature).squeeze(2)
            #print ('pooled art:',pooled_art.size(),'out art:',out_art.size())
            #print (attn_pooled_art.size())
        # Embed word ids to vectors
        #print (x[0].size(),x[1].size())
        #print (lengths[0],lengths[1])
        #print ('embed size:',x.size())
        #x = self.cnn_reduce_size(x)
        #print ('after conv size:',x.size())

        # textcnn
        #out = self.cnn_enc(x)

        ''' TCN
        out = self.cnn_enc(x.transpose(1,2))
        #print (out.size())
        out = self.pool(out).transpose(1,2)
        #print (out.size())
        out = out.squeeze(1)
        #print (out.size())
        '''
        ''' cnn 0.0
        x = self.m1(x.transpose(1,2))
        x = self.m2(x)
        x = self.m3(x)
        x = x.transpose(1,2)
        out = x.view(len(lengths),-1)
        '''
    
        # RNNs
        #'''# for GRU/LSTM
        #print ('after conv1d size:',x.size())
        #print ('lengths[0]:',lengths[0])
        #lengths = [l/2 for l in lengths] # lengths are now [64,...,64]

        #packed_cap = pack_padded_sequence(x_cap, lengths[0], batch_first=True)
        #packed_art = pack_padded_sequence(x_art, lengths[1], batch_first=True) 
        #print ('packed:',Variable(packed).size())

        # Forward propagate RNN
        #out_cap, _ = self.rnn(x_cap)
        #out_art, _ = self.rnn_art(x_art)
        #print ('out_cap:',out_cap.size(),'out_art:',out_art.size())

        # Reshape *final* output to (batch_size, hidden_size)
        #padded_cap = pad_packed_sequence(out_cap, batch_first=True)
        #padded_art = pad_packed_sequence(out_art, batch_first=True)
        #print (padded[0].size())
        #I = torch.LongTensor(lengths).view(-1, 1, 1)
        #I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        #out = torch.gather(padded[0], 1, I).squeeze(1)
        #out_cap = self.pool_cap(out_cap.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        #out_art = self.pool_art(out_art.transpose(1,2)).squeeze(2) # use pooling rather than nn.gather
        #print (out_cap.size(),out_art.size())
        #print (out.size())
        #del padded_cap
        #del padded_art
        #del packed_cap
        #del packed_art
        #del I
        #'''# for GRU/LSTM end

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
    
        return out

    
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            
        return cost_s.sum() + cost_im.sum()


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt, vocab, embeddings):
        # tutorials/09 - Image Captioning
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        #vocab = pickle.load(open(opt.vocab_path,'rb')) 
        self.vocab = vocab
        self.opt = opt
        self.label = opt.label
        self.text_encoder = opt.text_encoder
        
        try:
            _ = opt.pretrained_emb
        except:
            opt.pretrained_emb = False
        try:
            _ = opt.freeze_emb
        except:
            opt.freeze_emb = False
        try:
            _ = opt.test
        except:
            # change to true when testing
            opt.test = False
        resume = None
        try:
            if opt.resume != '':
                resume = True
        except:
            resume = False
        both = False
        if self.opt.label == 'both':
            both = True

        self.txt_enc = None
        if opt.text_encoder == 'default':
            self.txt_enc = EncoderText(vocab,opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs,
                                   pretrained_emb=opt.pretrained_emb,
                                   freeze_emb=opt.freeze_emb,
                                   test = opt.test,
                                   resume = resume,
                                   both=both,
                                   embeddings=embeddings
                                   )
        elif opt.text_encoder in ['caption','article','title','lead']:
            self.txt_enc = EncoderTextUnimodal(vocab,opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs,
                                   pretrained_emb=opt.pretrained_emb,
                                   freeze_emb=opt.freeze_emb,
                                   test = opt.test,
                                   resume = resume,
                                   both=both,
                                   embeddings=embeddings,
                                   label=opt.text_encoder
                                   )
        elif opt.text_encoder == 'transformer' and opt.label != "joint":
            print ('[using Transormer as text encoder!]')
            self.txt_enc = EncoderTextTransformer(vocab, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs,
                                   pretrained_emb=opt.pretrained_emb,
                                   freeze_emb=opt.freeze_emb,
                                   test = opt.test,
                                   resume = resume,
                                   both=both,
                                   label=opt.label,
                                   embeddings=embeddings,
                                   lang=opt.lang
                                   )
        elif opt.text_encoder == 'transformer' and opt.label == "joint":
            print ('[using Multimodal Transormer as text encoder!]')
            self.txt_enc = EncoderTextMultimodalTransformer(vocab,opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs,
                                   pretrained_emb=opt.pretrained_emb,
                                   freeze_emb=opt.freeze_emb,
                                   test = opt.test,
                                   resume = resume,
                                   both=both,
                                   label=opt.label,
                                   embeddings=embeddings
                                   )

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        
        self.txt_enc.requires_grad = True
        
        # to avoid pass in freezed params
        params = filter(lambda p: p.requires_grad, params)

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, text_inputs, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = torch.Tensor(images).cuda()
        texts = None
        if self.label in ['caption','article','lead','title']:
            texts_i = [inputs.cuda() for inputs in text_inputs[0]]
            texts_o = [offsets.cuda() for offsets in text_inputs[1]]
            texts = (texts_i,texts_o)
        elif self.label == "joint":
            #for row in text_inputs[0]:
            caps_i = [inputs.cuda() for inputs in text_inputs[0]]
            caps_o = [offsets.cuda() for offsets in text_inputs[1]]
            arts_i = [inputs.cuda() for inputs in text_inputs[2]]
            arts_o = [offsets.cuda() for offsets in text_inputs[3]]
            tits_i = [inputs.cuda() for inputs in text_inputs[4]]
            tits_o = [offsets.cuda() for offsets in text_inputs[5]]
            leds_i = [inputs.cuda() for inputs in text_inputs[6]]
            leds_o = [offsets.cuda() for offsets in text_inputs[7]]
            texts = [(caps_i,caps_o),(arts_i,arts_o),(tits_i,tits_o),(leds_i,leds_o)]
            #texts = [(caps_i,caps_o),(arts_i,arts_o)]


        #captions_ = [Variable(captions[0]),Variable(captions[1]),Variable(captions[2]),Variable(captions[3])]

        # Forward
        #import ipdb; ipdb.set_trace()
        cap_emb,img_emb = None,None
        if volatile == True:
            with torch.no_grad():
                img_emb = self.img_enc(images)
                cap_emb = self.txt_enc(texts)
        else:
            img_emb = self.img_enc(images)
            cap_emb = self.txt_enc(texts)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('loss', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, texts, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, texts)

        if isinstance(cap_emb,tuple):
            cap_emb = cap_emb[0]

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        del loss
        del img_emb
        del cap_emb
