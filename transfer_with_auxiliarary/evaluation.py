from __future__ import print_function
import os
import sys
import pickle
import numpy
import argparse
from data import *
import time
import numpy as np
import faiss                   # make faiss available
from vocab import Vocabulary  # NOQA
import torch
from torch.autograd import Variable
from model import VSE, order_sim
from collections import OrderedDict
from text_preprocessing import TextPreprocessing

preprocess = TextPreprocessing(
        twitter=True, replace_number=True, clean_html=True,
        capitalize=False, repeat=False, elong=False,
        replace_url=True, replace_emoticon=True,
        lower=True, lang="de")

def faiss_index(img_matrix):
    img_dimension = img_matrix.shape[1] 
    index = faiss.IndexFlatIP(img_dimension)   # build the index based on IP: inner product (cosine)
    # add img emb into faiss index
    index.add(img_matrix.astype('float32')) 
    return index

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    #print ("data_loader.dataset length:",len(data_loader.dataset))
    for i, (images, captions, ids) in enumerate(data_loader):
        
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, volatile=True)

        if isinstance(cap_emb,tuple):
            # if is tuple, then it contains attn scores
            cap_emb = cap_emb[0]

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids[0]:ids[-1]+1] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids[0]:ids[-1]+1] = cap_emb.data.cpu().numpy().copy()

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)

    # construct model
    model = VSE(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.crop_size,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)


    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure=opt.measure,
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure=opt.measure,
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])
    print ('images.shape:',images.shape)
    print ('captions.shape:',captions.shape)
    print ('npts:',npts)
    print ('ims.shape:',ims.shape)

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        print ("queries.shape:",queries.shape)

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

        print ("d.shape:",d.shape)

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)



def i2t_article(images, captions, caption_path, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] 
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    # for computing BLEU scores
    retrieve_num = 30
    f_index = faiss_index(np.array(captions))
    df = pd.read_csv(open(caption_path,'rb'))
    total_score = .0
    counter = 0
    BLEU_at_rank = np.zeros(30)
    
    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])
        
        scores,indicies = f_index.search(np.array([images[index]]).astype('float32'),retrieve_num) # calculate first <retrieve_num>
        #print (scores[0])
        #print (indicies[0])
        # get retrieved ids
        ids = [df.iloc[int(indicies[0][i])][['id']].tolist()[0] for i in range(retrieve_num)]
        # get ground truth text
        #gt_row = df[df['id'] == ids[index]]
        gt_row = df.iloc[index]
        #print (gt_row)
        gt_text = gt_row[['caption']].iloc[0]
        # find all captions
        '''
       for i,id in enumerate(ids):

            row = df[df['id'] == id]
            retrieved_cap_text = row[['caption']].iloc[0][0]
            # compute score
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [retrieved_cap_text.split(' ')],
                    gt_text.split(' '),
                    weights = (0.5,0.5,0,0))
            total_score += bleu_score
            BLEU_at_rank[i] += bleu_score
            counter += 1
        '''
        

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(index,index + 1, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr), (0,0) # avg_BLEU,BLEU_at_rank


def t2i_article(images, captions, caption_path, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] 

    ims = images

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    # for computing BLEU scores
    retrieve_num = 30
    f_index = faiss_index(np.array(images))
    df = pd.read_csv(open(caption_path,'rb'))
    total_score = .0
    counter = 0
    BLEU_at_rank = np.zeros(retrieve_num)
    
    for index in range(npts):

        # Get query captions
        queries = captions[index:index+1]
        #print (queries.shape)

        scores,indicies = f_index.search(np.array([captions[index]]).astype('float32'),retrieve_num) # calculate first 10
        #print (scores[0])
        #print (indicies[0])
        # get retrieved ids
        ids = [df.iloc[int(indicies[0][i])][['id']].tolist()[0] for i in range(retrieve_num)]
        # get ground truth text
        #gt_row = df[df['id'] == ids[index]]
        gt_row = df.iloc[index]
        #print (gt_row)
        gt_text = gt_row[['caption']].iloc[0]
        # find all captions
        '''
        for i,id in enumerate(ids):

            row = df[df['id'] == id]
            retrieved_cap_text = row[['caption']].iloc[0][0]
            # compute score
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [retrieved_cap_text.split(' ')],
                    gt_text.split(' '),
                    weights = (0.5,0.5,0,0))
            total_score += bleu_score
            BLEU_at_rank[i] += bleu_score
            counter += 1
        '''

        # Compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[index + i] = numpy.where(inds[i] == index)[0][0]
            top1[index + i] = inds[i][0]
        
        #if index in {0,1,2,3}:
        #    print (index,":",inds[0])

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr), (0,0) #avg_BLEU


# Fangyu @ EPFL
# July 2nd, 2018

def compute_img_embs(model, img_folder_path, caption_path, vocab_path):
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.vocab_path = vocab_path
    print (opt)
    # load vocabulary used by the model
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)
    opt.val_image_path = img_folder_path
    opt.val_caption_path = caption_path
    opt.test = True

    #weights = torch.load_state_dict(checkpoint['model'])
    #print (weights)
    #sys.exit(0)

    model = VSE(opt,vocab,None)
    model.load_state_dict(checkpoint['model'])
    print ('model loaded')
    
    print('Loading dataset')
    _, val_loader,_ = get_loaders(
            'article', vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    img_embs, _ = encode_data(model, val_loader)
    print ("img embedding shape:",img_embs.shape)
    return img_embs


def tokens_vector_representation(vocab, tokens):
    cap_matrix = []
    cap_matrix.append(vocab('<start>'))
    cap_matrix.extend([vocab(token) for token in tokens])
    cap_matrix.append(vocab('<end>'))
    cap_matrix = torch.Tensor(cap_matrix)
    # extend dim
    caps = torch.zeros(1,len(cap_matrix)).long()
    caps[0,:len(cap_matrix)] = cap_matrix[:len(cap_matrix)]
    if torch.cuda.is_available():
        caps = caps.cuda()
    return caps



def image_retriever(model_path, title,lead,caption,article,image_embs, index, caption_path, vocab_path, retrieve_num=20):
    """
    text to image (search image with text)
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.vocab_path = vocab_path
    opt.test = True
    print (opt)

    # load vocabulary used by the model
    with open(opt.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)
    # construct model
    model = VSE(opt)

    # load  model state
    model.load_state_dict(checkpoint['model'])
    
    title_tokens = preprocess(str(title))
    lead_tokens = preprocess(str(lead))
    cap_tokens = preprocess(str(caption))
    art_tokens = preprocess(str(article))
    #print ('query caption:',caption)
    #print ('tokens:',tokens)
    titles = tokens_vector_representation(vocab, title_tokens)
    leads = tokens_vector_representation(vocab, lead_tokens)
    caps = tokens_vector_representation(vocab, cap_tokens)
    arts = tokens_vector_representation(vocab, art_tokens)

    print('Computing text embedding...')
    #with torch.no_grad():
    fused_embs = model.txt_enc([Variable(caps).cuda(),
                                Variable(arts).cuda(),
                                Variable(titles).cuda(),
                                Variable(leads).cuda()],
                                [len(caps[0]),len(arts[0]),len(titles[0]),len(leads[0]) ])
    
    # build faiss index
    #print('building index...')
    #index = faiss_index(image_embs)
    '''
    img_dimension = image_embs.shape[1] 
    index = faiss.IndexFlatIP(img_dimension)   # build the index based on IP: inner product (cosine)
    # add img emb into faiss index
    index.add(image_embs.astype('float32')) 
    '''

    # Get query captions
    queries = fused_embs.data.cpu().numpy()
    print ('query shape:',queries.shape)
    # search
    print('searching...')
    scores,indicies = index.search(queries,retrieve_num)
    #print (scores[0])
    #print (indicies[0])
    
    # get ids
    caption_file = pd.read_csv(open(caption_path,'rb'))
    ids = [caption_file.iloc[int(indicies[0][i])][['id']].tolist()[0] for i in range(retrieve_num)]
    return ids, scores[0]


def Flask_wrapper(test_img_emb_path,
        model_path, title, lead, caption, article, 
        test_caption_file, vocab_path,retrieve_num=20):
    print ('load image embeddings...')
    img_embs = np.load(test_img_emb_path)
    print ('img emb shape:',img_embs.shape)
    print ('building faiss index...')
    index = faiss_index(img_embs)
    print ('searching images...')
    ids, sims = image_retriever(model_path, title,lead,caption,article,img_embs,index,test_caption_file,vocab_path,retrieve_num)
    print ('top ids:',ids)
    print ('top sims:',sims)
    #numpy.savetxt("/home/fangyu/data/ids_tmp.csv",ids,delimiter=',')
    #numpy.savetxt("/home/fangyu/data/sims_tmp.csv",sims,delimiter=',')
    info_list = []
    df = pd.read_csv(test_caption_file)
    for index,id in ids:
        ''' find (image_name, sim_score, title, lead, caption, article) and append to info list'''
        row = df[df['id'] == id]
        img_path = row[['img_name']].iloc[0][0]
        title = row[['title']].iloc[0][0]
        lead = row[['lead']].iloc[0][0]
        caption = row[['caption']].iloc[0][0]
        article = row[['article']].iloc[0][0]
        info_list.append([img_path,sims[index],title,lead,caption,article])
    return info_list



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',default='/home/fangyu/data/models_for_demo/model_best.pth.tar',help='the .pth.tar file')
    parser.add_argument('--test_img_folder',default='/home/fangyu/repos/Flask_demo/static/images-full/',help='')
    parser.add_argument('--test_caption_file',default='/home/fangyu/data/full_title_lead_caption_article.csv',help='')
    parser.add_argument('--title',default='donald trump and xi jinping',help='')
    parser.add_argument('--lead',default='donald trump and xi jinping',help='')
    parser.add_argument('--caption',default='donald trump and xi jinping',help='')
    parser.add_argument('--article',default='donald trump and xi jinping',help='')
    parser.add_argument('--compute_test_img_embeddings',action='store_true')
    parser.add_argument('--test_img_emb_path',
            default='/home/fangyu/data/img_embs.npy',type=str)
    parser.add_argument('--vocab_path',default='vocab/images-12345678910-threshold-3.pkl',type=str)
    opt = parser.parse_args()
    
    model_path = opt.model_path
    test_img_folder = opt.test_img_folder
    test_caption_file = opt.test_caption_file
    title = opt.title
    lead = opt.lead
    caption = opt.caption
    article = opt.article

    img_embs = None
    if opt.compute_test_img_embeddings:
        print ('computing image embeddings...')
        img_embs = compute_img_embs(model_path, test_img_folder, test_caption_file, opt.vocab_path)
        np.save('./img_embs.npy',img_embs)
    else:
        print ('load image embeddings...')
        img_embs = np.load(opt.test_img_emb_path)
        print ('img emb shape:',img_embs.shape)
    """
    print ('building faiss index...')
    index = faiss_index(img_embs)
    print ('searching images...')
    ids, sims = image_retriever(model_path, title,lead,caption,article,img_embs, index, test_caption_file, opt.vocab_path)
    print ('top ids:',ids)
    print ('top sims:',sims)
    numpy.savetxt("/home/fangyu/data/ids_tmp.csv",ids,delimiter=',')
    numpy.savetxt("/home/fangyu/data/sims_tmp.csv",sims,delimiter=',')
    """

    
