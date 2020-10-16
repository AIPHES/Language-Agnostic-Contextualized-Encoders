from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from pyemd import emd, emd_with_flow
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial

from transformers import (
        BertPreTrainedModel, 
        BertModel,
        BertConfig,
        BertTokenizer,
)

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.pooler = BertPooler(config)
        self.init_weights()
    
    def z_norm(self, inputs):
        mean = inputs.mean(0, keepdim=True)
        var = inputs.var(0, unbiased=False, keepdim=True)
        return (inputs - mean) / torch.sqrt(var + 1e-9)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        is_norm=None,
        layer=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[2][layer]
        if is_norm:
            sequence_output = self.z_norm(sequence_output)
        
        pooled_output = self.pooler(sequence_output)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


from transformers import DataProcessor

import copy
import csv
import json
import logging
from transformers import XLMTokenizer

class InputExample(object):

  def __init__(self, guid, text_a, text_b=None, label=None, language=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.language = language

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

import os
class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
      """See base class."""
      examples = []
      for lg in language.split(','):
        lines = self._read_tsv(os.path.join(data_dir, "xnli.translate.pseudo-test.en-{}.tsv".format(lg)))
        
        for (i, line) in enumerate(lines):
          if i == 0:
            continue
          guid = "%s-%s-%s" % (split, lg, i)
          text_a = line[0]
          text_b = line[1]
          if split == 'test' and len(line) != 3:
            label = "neutral"
          else:
            label = str(line[2].strip())
          assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
          examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
      return examples

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]
    
class InputFeatures(object):
  """
  A single set of features of data.
  Args:
    input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    label: Label corresponding to the input
  """

  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

from transformers import glue_convert_examples_to_features as convert_examples_to_features

def load_and_cache_examples(tokenizer, language='en', lang2id=None, evaluate=False):

    processor = XnliProcessor()
    output_mode = "classification"

    label_list = processor.get_labels()
    examples = processor.get_test_examples('XNLI/translate-pseudo-test', language)
  
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=128,
        output_mode=output_mode,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset

def evaluate(layer, model, tokenizer, is_norm=False, eval_batch_size=8, split='train', language='en', lang2id=None, prefix="", output_file=None, label_list=None, output_only_prediction=True):
  """Evalute the model."""
  
 
  eval_dataset = load_and_cache_examples(tokenizer, language=language, lang2id=lang2id, evaluate=True)
  
  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
  
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to('cuda') for t in batch)
    
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = (batch[2])
            inputs["is_norm"] = is_norm
            inputs["layer"] = layer
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
  preds = np.argmax(preds, axis=1)
    
  result = compute_metrics(preds, out_label_ids)
    
  return result

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='de', type=str, help="target language")
parser.add_argument("--model_path", default='bert-base-m-checkpoint-0-LR2e-5-epoch3-MaxLen128', type=str)
parser.add_argument("--layer", default='-1', type=int, help='in which layer embeddings are obtained')

args = parser.parse_args()

config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)
tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=False)

model = model_class.from_pretrained(args.model_path, output_hidden_states=True)
model.to('cuda')

result_1 = evaluate(args.layer, model, tokenizer, is_norm=False, eval_batch_size=8, split='test', language=args.lang)['acc']
result_2 = evaluate(args.layer, model, tokenizer, is_norm=True, eval_batch_size=8, split='test', language=args.lang)['acc']

print('layer:{} {}->{}'.format(args.layer,'en', args.lang), '{}->{}'.format('{0:.{1}f}'.format(result_1,3), '{0:.{1}f}'.format(result_2,3)))
