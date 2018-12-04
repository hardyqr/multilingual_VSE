import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import math
import sys
import time
import shutil
import torch
import data
from vocab import Vocabulary  # NOQA
print ('vocab imported')
from model import VSE
from evaluation import i2t, t2i, i2t_article,t2i_article,AverageMeter, LogCollector, encode_data
import logging
print ('good so far')
import tensorboard_logger as tb_logger
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import *

# torch-goodies
from embeddings import FastTextVocab
from embeddings import FastTextEmbeddings

print ('succesful imported')

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='article_db',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k|article_db')
    parser.add_argument('--vocab_path', default='./vocab/xxx.pkl',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--vocab_de', default='./vocab/xxx.pkl')
    parser.add_argument('--vocab_fr', default='./vocab/xxx.pkl')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=400, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=14, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=800, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', 
                        default='/mnt/storage01/fangyu/vsepp_models/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='vgg19',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--caption_path_de', default='./',type=str,
                        help='img caption csv file path')
    parser.add_argument('--caption_path_fr', default='./',type=str)
    parser.add_argument('--image_path', default='./',type=str,
                        help='img folder path')
    parser.add_argument('--val_caption_path_de', default='.',type=str,
                        help='val img caption csv file path')
    parser.add_argument('--val_caption_path_fr', default='.',type=str)
    parser.add_argument('--val_image_path', default='./',type=str,
                        help='val img folder path')
    parser.add_argument('--pretrained_emb', action='store_true')
    parser.add_argument('--freeze_emb', action='store_true')
    parser.add_argument('--train_translator_only', action='store_true')
    parser.add_argument('--places365resnet50',action='store_true')
    parser.add_argument('--epoch_free_cnn', default=5,type=int,
                        help='')
    parser.add_argument('--label', default='caption',type=str,
                        help='val img folder path')
    parser.add_argument('--text_encoder', default='default',type=str,
                        help='{default|ulmfit|...}')
    parser.add_argument('--random_miss', action='store_true')
    parser.add_argument('--freeze_translator', action='store_true')
    parser.add_argument('--LM_path', default='',type=str)
    parser.add_argument('--lang', default='de',type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--finetune_enc', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--unfreeze_word_emb', default=30,type=int)
    parser.add_argument('--unfreeze_all', default=4,type=int)

    opt = parser.parse_args()
    print(opt)
    
    lang = opt.lang

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab_de = pickle.load(open(opt.vocab_de,'rb'))
    vocab_fr = pickle.load(open(opt.vocab_fr,'rb'))

    
    ft_vocab = [vocab_de,vocab_fr] # if directly load
    #ft_vocab = pickle.load(open('./vocab/ft_vocab_article_min4_max_6_1_3.pkl','rb'))
    #ft_vocab = pickle.load(open('./vocab/ft_vocab_min4_max_6_1_3_cc.pkl','rb'))
   
    # Load data loaders
    print ('--- loading data ---')
    opt.val_caption_path = opt.val_caption_path_de
    opt.caption_path = opt.caption_path_de
    _, val_loader_de, _ = data.get_loaders(
        opt.data_name, ft_vocab, opt.crop_size, opt.batch_size, opt.workers, opt)
    opt.val_caption_path = opt.val_caption_path_fr
    opt.caption_path = opt.caption_path_fr
    _, val_loader_fr, _ = data.get_loaders(
        opt.data_name, ft_vocab, opt.crop_size, opt.batch_size, opt.workers, opt)   

    val_loaders = [val_loader_de, val_loader_fr]

    embeddings = None
    # Construct the model
    print ('--- constructing model ---')
    model = VSE(opt,ft_vocab,embeddings)


    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']

            #""" # use when transfer across lang
            # change transformer.src_word_emb
            state_dict_enc = checkpoint["model"][1]
            # substitue with fr embedding
            n_v_de = list(torch.load('vocab_de_4_6_1_3_embedding_bag_.pth').items())
            print ('de word embed: {}'.format(n_v_de[0][1].size()))
            n_v_fr = list(torch.load('vocab_fr_4_6_1_3_embedding_bag_.pth').items())
            print ('fr word embed: {}'.format(n_v_fr[0][1].size()))
            #print ("n_v.size: {}".format(n_v.size()))
            #state_dict_enc = {k:v if "src_word_emb" not in k else n_v[0][1] for k,v in state_dict_enc.items()}
            #state_dict_enc['transformer.src_word_emb.weight'] = model.txt_enc.transformer.src_word_emb.weight.data # keep the fr embeddings weight
            state_dict_enc['transformer.src_word_emb_de.weight'] = n_v_de[0][1]
            state_dict_enc['transformer.src_word_emb_fr.weight'] = n_v_fr[0][1]
            
            state_dict_enc.pop('transformer.src_word_emb.weight', None)

            # add translator
            #translator_weight = torch.tensor(load_matrix('/home/fangyu/repos/fastText/alignment/res/subword.fr-de.googleAPI.vec-mat'),dtype=torch.float)#.t() # transpose cuz originaly np.dot(input,m), now F.linear is xA^T
            #m = np.linalg.inv(load_matrix('/home/fangyu/repos/fastText/alignment/res/subword.fr-de.vec-mat.prev.sota'))
            m = load_matrix('subword.fr-de.vec-mat')
            translator_weight = torch.tensor(m,dtype=torch.float) # transpose cuz originaly np.dot(input,m), now F.linear is xA^T
            #translator_weight = torch.FloatTensor((300,300)).uniform_(-0.1, 0.1)# random init
            #translator_weight = torch.randn((300,300)) # random init
            #"\"" # random translator weights
            #tmp = nn.Linear(300,300)
            #torch.nn.init.xavier_uniform(tmp.weight) # it's better without xavier
            #translator_weight = tmp.weight.data
            #"\""
            stdv = 1. / math.sqrt(translator_weight.size(1))
            #stdv = 0.1
            #""" # commented to test no translator situation
            state_dict_enc['transformer.translator.weight'] = translator_weight
            state_dict_enc['transformer.translator.bias'] = torch.Tensor(300).uniform_(-stdv,stdv)
            #"""
            
            #print (state_dict_enc['transformer.src_word_emb.weight'].size())

            checkpoint['model'][1] = state_dict_enc
            #"""

            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            #validate(opt, val_loader, model)
            #model.Eiters = 0
            #print ("model.Eithers := 0")
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.train_translator_only == True:
        model.txt_enc.requires_grad = False
        model.txt_enc.transformer.translator.requires_grad = True # only train translator!
        print ('[only training translator in txt_enc!]')
    else: 
        #model.txt_enc.requires_grad = True
        #print ('[training the whole txt_enc!]')
        print ('[not only train translator]')

    if opt.freeze_translator == True:
        #model.txt_enc.requires_grad = True
        model.txt_enc.transformer.translator.requires_grad = False # only train translator!
        print ('[translator freezed!]')
    else: 
        #model.txt_enc.requires_grad = True
        model.txt_enc.transformer.translator.requires_grad = True # only train translator!
        print ('[training the translator!]')

    if opt.finetune_enc and opt.label == "joint" :
        model.txt_enc.transformer_cap.requires_grad = True
        model.txt_enc.transformer_art.requires_grad = True
        model.txt_enc.transformer_tit.requires_grad = True
        model.txt_enc.transformer_led.requires_grad = True
        print ("training transformers.")
    elif not opt.finetune_enc and opt.label == "joint":
        model.txt_enc.transformer_cap.requires_grad = False
        model.txt_enc.transformer_art.requires_grad = False
        model.txt_enc.transformer_tit.requires_grad = False
        model.txt_enc.transformer_led.requires_grad = False
        print ("transformers freezed.")

    if opt.freeze_emb == True:
        model.txt_enc.transformer.src_word_emb_de.require_grad = False
        model.txt_enc.transformer.src_word_emb_fr.require_grad = False
        print ('[fr word embed freezed]')
    else:
        model.txt_enc.transformer.src_word_emb_de.require_grad = False
        model.txt_enc.transformer.src_word_emb_fr.require_grad = True
        print ('[fr word embed training]')
    # val in the beginning
    #'''
    #print ('******* full val *******')
    if opt.evaluate:
        rsum = validate(opt, val_loaders[0], model)
        sys.exit(0)


    # Train the Model
    best_rsum = 0

    df_1 = pd.read_csv(open(opt.caption_path_de,'rb'),index_col=False) 
    df_2 = pd.read_csv(open(opt.caption_path_fr,'rb'),index_col=False)
    
    df_new = active_sampling(df_1,df_2,1,1.07) # 1.07 for 2:1 sampling
    df_new.to_csv('tmp.csv')
    opt.caption_path = 'tmp.csv'
    train_loader,_,_ = data.get_loaders(
        opt.data_name, ft_vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    print ('--- training ---')
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loaders)

        # evaluate on validation set
        print ('******* full val *******')
        rr1 = validate(opt, val_loaders[0], model)
        rr2 = validate(opt, val_loaders[1], model)
        r1 = (rr1[0]+rr1[1])/2/100
        r2 = (rr2[0]+rr2[1])/2/100

        r1,r2 = 1,1.07 #1.07
        print (r1,r2)
        
        #df_new = active_sampling(df_1,df_2,r1,r2)
        df_new = active_sampling(df_1,df_2,r1,r2)
        df_new.to_csv('tmp.csv')
        opt.caption_path = 'tmp.csv'
        train_loader,_,_ = data.get_loaders(
        opt.data_name, ft_vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

        """
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')
        """


def calc_ratio(r1,r2,tau=0.1):
    return np.exp(-r1/tau)/(np.exp(-r1/tau)+np.exp(-r2/tau))

def active_sampling(df_1,df_2,r1,r2):
    """
    it is required that len(df_1) == len(df_2)
    r1 - recall @10 on de
    r2 - recall @10 on fr
    """
    ratio = calc_ratio(r1,r2)
    num_1 = int(len(df_1)*ratio)
    num_2 = int(len(df_1)*(1-ratio))
    df_1 = df_1.sample(n=num_1)
    df_2 = df_2.sample(n=num_2)
    df_new = df_1.append(df_2)
    print ('{} de, {} fr'.format(num_1,num_2))
    print ('new_df size {}'.format(len(df_new)))
    return df_new


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    print ('---reset to train mode---')
    model.train_start()

    end = time.time()

    # unfreeze txt encoder
    if epoch == opt.unfreeze_word_emb:
        model.txt_enc.transformer.src_word_emb_de.requires_grad = True
        print ('[start finetuning de word embed!]')
    if epoch == opt.unfreeze_all:
        model.txt_enc.transformer.requires_grad = True
        print ('[start finetuning everything!]')



    for i, train_data in enumerate(train_loader):
        #if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
        #    model.train_start()
        
        # added by me, always reset to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        #if model.Eiters % opt.val_step == 0:
        #    r1,r2 = validate(opt, val_loader, model)
    del train_loader



def validate(opt, val_loader, model,text=''):

    model.val_start()

    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr),(i2t_b_score,i2t_BLEU_at_rank) = i2t_article(img_embs, cap_embs, opt.val_caption_path, measure=opt.measure)
    #(r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs,measure=opt.measure)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, top10_avg_BLEU: %.4f" %
                 (r1, r5, r10, medr, meanr,i2t_b_score))
    # image retrieval
    #(r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, measure=opt.measure)
    (r1i, r5i, r10i, medri, meanr),(t2i_b_score,t2i_BLEU_at_rank) = t2i_article(img_embs, cap_embs, opt.val_caption_path, measure=opt.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, top10_avg_BLEU: %.4f" %
                 (r1i, r5i, r10i, medri, meanr,t2i_b_score))

    # print BLEU@rank
    #npts = len(i2t_BLEU_at_rank) 
    #xs = [i for i in range(1,npts+1)]
    #print ('i2t avg_BLEU@rank:',i2t_BLEU_at_rank)
    #print ('t2i avg_BLEU@rank:',t2i_BLEU_at_rank)
    '''# plot avg_BLEU@rank
    plt.plot(xs,i2t_BLEU_at_rank,'o-',label='i2t')
    #plt.ylabel('i2t avg_BLEU@rank')
    #plt.savefig('i2t_avg_BLEU_at_rank_'+str(model.Eiters)+'.png')
    plt.plot(xs,t2i_BLEU_at_rank,'o-',label='t2i')
    plt.ylabel('avg_BLEU')
    plt.xlabel('rank')
    plt.savefig('avg_BLEU_at_rank_'+str(model.Eiters)+'.png')
    plt.close()
    '''

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1'+text, r1, step=model.Eiters)
    tb_logger.log_value('r5'+text, r5, step=model.Eiters)
    tb_logger.log_value('r10'+text, r10, step=model.Eiters)
    tb_logger.log_value('medr'+text, medr, step=model.Eiters)
    tb_logger.log_value('meanr'+text, meanr, step=model.Eiters)
    #tb_logger.log_value('i2t_top10_avg_bleu_score'+text, i2t_b_score, step=model.Eiters)
    tb_logger.log_value('r1i'+text, r1i, step=model.Eiters)
    tb_logger.log_value('r5i'+text, r5i, step=model.Eiters)
    tb_logger.log_value('r10i'+text, r10i, step=model.Eiters)
    tb_logger.log_value('medri'+text, medri, step=model.Eiters)
    tb_logger.log_value('meanr'+text, meanr, step=model.Eiters)
    tb_logger.log_value('rsum'+text, currscore, step=model.Eiters)
    #tb_logger.log_value('t2i_top10_avg_bleu_score'+text, t2i_b_score, step=model.Eiters)

    #return currscore
    return (r10,r10i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 5 every xx epochs"""
    lr = opt.learning_rate * (0.2 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
