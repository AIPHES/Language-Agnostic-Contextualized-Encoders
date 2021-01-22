from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from pyemd import emd
from collections import defaultdict
from transformers import *

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)
        x_encoded_layers = outputs[2]
    return x_encoded_layers
       
def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device='cuda:0'):
    tokens = [["[CLS]"]+tokenize(a)+["[SEP]"] for a in arr]

    arr = [numericalize(a) for a in tokens]
    
    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    
    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device='cuda:0'):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(all_sens,
                                                      tokenizer.tokenize, tokenizer.convert_tokens_to_ids,
                                                      idf_dict,
                                                      device=device)

    if batch_size == -1: batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            batch_embedding = bert_encode(model, padded_sens[i:i+batch_size],
                                          attention_mask=mask[i:i+batch_size])
            batch_embedding = torch.stack(batch_embedding)
            
            input_mask = mask[i:i+batch_size]
            input_mask_expanded = input_mask.unsqueeze(-1).expand(batch_embedding.size()).float()
            batch_embedding = batch_embedding * input_mask_expanded
        
            embeddings.append(batch_embedding)
            del batch_embedding

    total_embedding = torch.cat(embeddings, dim=-3)
    return total_embedding, lens, mask, padded_idf, tokens

def z_norm(inputs):
    mean = inputs.mean(0, keepdim=True)
    var = inputs.var(0, unbiased=False, keepdim=True)
    return (inputs - mean) / torch.sqrt(var + 1e-9)


def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)


def wmd(arguments):
    size, ref_idf, hyp_idf, dst = arguments
    
    c1 = np.zeros(size, dtype=np.float)
    c2 = np.zeros_like(c1)
    
    c1[:len(ref_idf)] = ref_idf
    c2[len(ref_idf):] = hyp_idf
    
    c1 = _safe_divide(c1, np.sum(c1))
    c2 = _safe_divide(c2, np.sum(c2))

    score = emd(c1, c2, np.asarray(dst, dtype='float64'))
    
    similarity = 1./(1. + score)
    return similarity

def optimal_score(layer, refs, hyps, is_norm=False, batch_size=256, device='cuda:0'):
    scores = []
    idf_dict_ref = defaultdict(lambda: 1.)
    idf_dict_hyp = defaultdict(lambda: 1.)
  
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]        
        
        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, model, tokenizer, idf_dict_ref,
                                       device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict_hyp,
                                       device=device)   
        
        ref_embedding = ref_embedding[layer]
        hyp_embedding = hyp_embedding[layer]
        
        if is_norm:
            ref_embedding = z_norm(ref_embedding)
            hyp_embedding = z_norm(hyp_embedding)
        
        raw = torch.cat([ref_embedding, hyp_embedding], 1)
                             
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
        
        distance_matrix = batched_cdist_l2(raw, raw).cpu().numpy().astype('float64')
              
        for i in range(batch_size):  
            c1 = np.zeros(raw.shape[1], dtype=np.float)
            c2 = np.zeros_like(c1)
            c1[:len(ref_idf[i])] = ref_idf[i]
            c2[len(ref_idf[i]):] = hyp_idf[i]
            
            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            
            score = emd(c1, c2, distance_matrix[i])
            scores.append(1./(1. + score))

    return scores

import pandas as pd 

import truecase  
from scipy.stats import pearsonr

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    return '{0:.{1}f}'.format(pearson_corr, 3)

from utils import remove_word_contraction, clean_text, find_model_path
import spacy_udpipe
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src", default='fr', type=str, help="source language")
parser.add_argument("--tgt", default='en', type=str, help="target language")
parser.add_argument("--is_align", default=False, type=bool, help="whether or not joint-alignment is enabled")
parser.add_argument("--layer", default='-1', type=int, help='in which layer embeddings are obtained')

args = parser.parse_args()

spacy_udpipe.download(args.src)
spacy_udpipe.download(args.tgt)


model_path = find_model_path(args.src)

if not args.is_align:
    model_name = 'bert-base-multilingual-cased'
else:
    model_name = model_path

dataset_path = 'dataset/testset_{}-{}.tsv'.format(args.src, args.tgt)

config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name, config=config)
model.eval()
model.to('cuda')

data = pd.read_csv(dataset_path, sep='\t') 
translations = data['translation'].tolist()
source = data['source'].tolist()
human_score = data['HUMAN_score'].tolist()
sentBLEU = data['sentBLEU'].tolist()

from mosestokenizer import MosesDetokenizer
with MosesDetokenizer(args.src) as detokenize:        
   source = [detokenize(s.split(' ')) for s in source]         
with MosesDetokenizer(args.tgt) as detokenize:                
   translations = [detokenize(s.split(' ')) for s in translations]        

src_udpipe = spacy_udpipe.load(args.src)
tgt_udpipe = spacy_udpipe.load(args.tgt)

translations = [truecase.get_true_case(s) for s in translations]

source_manipulation, _ = remove_word_contraction(src_udpipe, source, args.src)    
translations_manipulation, _ = remove_word_contraction(tgt_udpipe, translations, args.tgt)

source = [clean_text(s, args.src) for s in source]
translations = [clean_text(s, args.tgt) for s in translations]
source_manipulation = [clean_text(s, args.src) for s in source_manipulation]
translations_manipulation = [clean_text(s, args.tgt) for s in translations_manipulation]

if not args.is_align:
    output_1 = optimal_score(args.layer, source, translations, is_norm=False, batch_size=8) # original
    output_2 = optimal_score(args.layer, source, translations, is_norm=True, batch_size=8)# norm_space
    output_3 = optimal_score(args.layer, source_manipulation, translations_manipulation, is_norm=False, batch_size=8) # norm_text
else:
    output_1 = optimal_score(args.layer, source, translations, is_norm=False, batch_size=8) # align
    output_2 = optimal_score(args.layer, source, translations, is_norm=True, batch_size=8)# align + norm_space
    output_3 = optimal_score(args.layer, source_manipulation, translations_manipulation, is_norm=True, batch_size=8) # align + norm_space + norm_text

corr_1 = pearson_and_spearman(human_score, output_1)
corr_2 = pearson_and_spearman(human_score, output_2)
corr_3 = pearson_and_spearman(human_score, output_3)

print('layer:{} {}->{}'.format(args.layer, args.src, args.tgt), '{}->{}->{}'.format(corr_1, corr_2, corr_3))
