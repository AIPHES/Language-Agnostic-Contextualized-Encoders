import pandas as pd 

import sys
import unicodedata
import string
import re

def find_model_path(src_lang):
    if src_lang in ['de','nl','es','fr','it','pt','et','fi','hu']:
        model_path = '../model/bert-base-m-cased_align_lan_part_1'
    
    if src_lang in ['hi','id','jv','tl','mr','ur','af','ms', 'bn']:
        model_path = '../model/bert-base-m-cased_align_lan_part_2'
    return model_path

PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)
WHITESPACE_LANGS = ['nl','jv','en','id','ko', 'ja','es', 'hi', 'vi', 'de', 'ar', 'fr', 'el', 'tr', 'ru', 'fi', 'cs', 'lv', 'it', 'eu','pt','ga', 'mr']
MIXED_SEGMENTATION_LANGS = ['zh']

def clean_text(s, lang):
    def whitespace_tokenize(text):
        return text.split() 
    
    def mixed_segmentation(text):
        segs_out = []
        temp_str = ""
        for char in text:
            if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
                if temp_str != "":
                    ss = whitespace_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char
    
        if temp_str != "":
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
    
        return segs_out
    
    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        if lang in ['zh','ja']:
            return ''.join([t for t in tokens if t.strip() != ''])    
        else:
            return ' '.join([t for t in tokens if t.strip() != ''])
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_punc(s), lang)

def remove_word_contraction(udpipe, sentences, lang):
    skip_count = 0
    return_sents = []
    for sent in sentences:
        tokens, feats = udpipe(sent)
        tmp = []
        for i, cur_token in enumerate(tokens):
            tmp.append(cur_token.text)
        if lang in ['zh', 'ja', 'th']:
            return_sents.append("".join(tmp))
        else:
            return_sents.append(" ".join(tmp))
    return return_sents, skip_count

    