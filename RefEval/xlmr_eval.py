from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from pyemd import emd
from collections import defaultdict
from transformers import *

def tokenize(text):
    """
    Tokenizes a text and maps tokens to token-ids
    """
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    
def get_sentence_features(tokens, pad_seq_length: int):
    """
    Convert tokenized sentence in its embedding ids, segment ids and mask
    :param tokens:
        a tokenized sentence
    :param pad_seq_length:
        the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
    :return: embedding ids, segment ids and mask for the sentence
    """
    pad_seq_length = min(pad_seq_length, max_seq_length) + 2 #Add space for special tokens
    return tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')

def encode(features):
    """Returns token_embeddings, cls_token"""
    #RoBERTa does not use token_type_ids
    output_states = model(**features)
    output_tokens = output_states[0]
    cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
    features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})

    if model.config.output_hidden_states:
        hidden_states = output_states[2]
        features.update({'all_layer_embeddings': hidden_states})

    return features

def get_word_embedding_dimension() -> int:
    return model.config.hidden_size

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def collate_idf(arr, tokenize, numericalize, 
                pad="[PAD]", device='cuda:0'):
    tokens = [["<s>"]+tokenize(a)+["</s>"] for a in arr]
    arr = [numericalize(a) for a in tokens]

    pad_token = 1
    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, lens, mask, tokens


def produce_tokens_masks(sent, max_length):
    input_ids = tokenizer.convert_tokens_to_ids(create_tokens(sent, None, max_length))
#    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)

    pad_token = 1
    padding_length = max_length - len(input_ids)         
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    return input_ids, attention_mask


def get_embedding(layer, sentences, batch_size= 8):
    
    padded_sens, lens, mask, tokens = collate_idf(sentences,
                                                      tokenizer.tokenize, 
                                                      tokenizer.convert_tokens_to_ids,
                                                      device='cuda')

    features = {"input_ids": padded_sens, "attention_mask": mask, "token_type_ids": None}
    with torch.no_grad():
        output_states = model(**features)
        all_embeddings = output_states[2][layer]
        input_mask = features['attention_mask']
        input_mask_expanded = input_mask.unsqueeze(-1).expand(all_embeddings.size()).float()
        all_embeddings = all_embeddings * input_mask_expanded
    return all_embeddings, mask

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


def optimal_score(layer, refs, hyps, is_norm=False, batch_size=256, device='cuda:0'):
    scores = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]        
        
        ref_embedding, ref_masks = get_embedding(layer, batch_refs, batch_size)
        hyp_embedding, hyp_masks = get_embedding(layer, batch_hyps, batch_size)   
        
        ref_idf = ref_masks.float().cpu()
        hyp_idf = hyp_masks.float().cpu()

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

from utils import remove_word_contraction, clean_text
import spacy_udpipe
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src", default='fr', type=str, help="source language")
parser.add_argument("--tgt", default='en', type=str, help="target language")
parser.add_argument("--is_align", default=True, type=bool, help="whether or not joint-alignment is enabled")
parser.add_argument("--model_path", default='../model/xlm-roberta-base_align_lang_18', type=str)
parser.add_argument("--layer", default='-1', type=int, help='in which layer embeddings are obtained')

args = parser.parse_args()

spacy_udpipe.download(args.src)
spacy_udpipe.download(args.tgt)

if not args.is_align:
    model_name = 'xlm-roberta-base'
else:
    model_name = args.model_path


dataset_path = 'dataset/testset_{}-{}.tsv'.format(args.src, args.tgt)

model = XLMRobertaModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, do_lower_case=False)

max_seq_length = tokenizer.max_len_single_sentence
device = 'cuda'
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
