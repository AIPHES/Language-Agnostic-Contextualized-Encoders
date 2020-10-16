import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SequentialSampler

from transformers import (
  XLMRobertaTokenizer,
)
from xlm_roberta import (
  XLMRobertaConfig,
  XLMRobertaForSequenceClassification
)

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


def truncate(tokens, max_len):
    if len(tokens) > max_len:
        tokens = tokens[0:max_len]
    return tokens

def create_tokens(example_a, example_b, max_length):
    sep_token = "</s>"
    cls_token = "<s>"
    cls = [cls_token]
    sep = [sep_token]
    if example_b is None:
        return cls + truncate(tokenizer.tokenize(example_a), max_length-2) + sep

    max_length = int((max_length-4)/2)
    return cls + truncate(tokenizer.tokenize(example_a), max_length) + sep + \
            sep + truncate(tokenizer.tokenize(example_b), max_length) + sep

def convert_examples_to_features(
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
):
  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):

    input_ids = tokenizer.convert_tokens_to_ids(create_tokens(example.text_a, example.text_b,128))
    token_type_ids = [0] * len(input_ids)

    attention_mask = [1] * len(input_ids)
    padding_length = max_length - len(input_ids)
    pad_token = 1
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
   
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
  
    label = label_map[example.label]

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
      )
    )
  return features

def compute_metrics(preds, labels):
  scores = {
    "acc": (preds == labels).mean(), 
    "num": len(
      preds), 
    "correct": (preds == labels).sum()
  }
  return scores

def evaluate(layer, model, tokenizer, is_norm=False, eval_batch_size=8, split='train', language='en', lang2id=None, prefix="", output_file=None, label_list=None, output_only_prediction=True):
  """Evalute the model."""
  
 
  eval_dataset = load_and_cache_examples(tokenizer, language=language, lang2id=lang2id, evaluate=True)
  
  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
  
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  sentences = None
  for batch in eval_dataloader:
    model.eval()
    batch = tuple(t.to('cuda') for t in batch)
  
    with torch.no_grad():
      inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids": None}
      inputs['is_norm'] = is_norm
      inputs["layer"] = layer
      
      outputs = model(**inputs)
      tmp_eval_loss, logits = outputs[:2]
  
    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
      sentences = inputs["input_ids"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
      sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)
  
  preds = np.argmax(preds, axis=1)
  
  result = compute_metrics(preds, out_label_ids)
  return result

from transformers import DataProcessor

import copy
import json

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


class XnliProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
      """See base class."""
      examples = []
      sents = []
      for lg in language.split(','):
        lines = self._read_tsv(os.path.join(data_dir, "xnli.translate.pseudo-test.en-{}.tsv".format(lg)))
        
        for (i, line) in enumerate(lines):
          if i == 0:
            continue
          guid = "%s-%s-%s" % (split, lg, i)
          text_a = line[0]
          text_b = line[1]
          sents.append(text_a + text_b)
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
    lang2id=lang2id,
  )
    
  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
#  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

  dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
  return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default='de', type=str, help="target language")
parser.add_argument("--model_path", default='xlm-roberta-base-LR2e-5-epoch5-MaxLen128', type=str)
parser.add_argument("--layer", default='-1', type=int, help='in which layer embeddings are obtained')

args = parser.parse_args()

config_class, model_class, tokenizer_class = (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer)
tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=False)

model = model_class.from_pretrained(args.model_path, output_hidden_states=True)
model.to('cuda')

result_1 = evaluate(args.layer, model, tokenizer, is_norm=False, eval_batch_size=8, split='test', language=args.lang)['acc']
result_2 = evaluate(args.layer, model, tokenizer, is_norm=True, eval_batch_size=8, split='test', language=args.lang)['acc']

print('layer:{} {}->{}'.format(args.layer,'en', args.lang), '{}->{}'.format('{0:.{1}f}'.format(result_1,3), '{0:.{1}f}'.format(result_2,3)))
