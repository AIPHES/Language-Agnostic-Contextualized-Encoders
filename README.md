
# Language-Agnostic Multilingual Representations
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python) 

This is a library to use our multilingual representations. The details are described in the paper [Inducing Language-Agnostic Multilingual Representations](https://arxiv.org/abs/2008.09112)


**CURRENT VERSION:**
* We provide the modifications to m-BERT and XLM-R, namely joint-align, space normalization and removing word contractions (other two text normalizations will be updated in next version).
* We provide the machine translated datasets on the RFEval and XNLI tasks.
* The pre-trained m-BERT and XLM-R models are re-aligned on word translations corpus extracted from Europarl and JW300, both of which support 19 languages: English, German, Portuguese, Dutch, Indonesian, Italian, French, Spanish, Hungarian, Afrikaans, Malay, Tagalog, Javanese, Bengali, Marathi, Estonian, Hindi, Urdu, Finnish. 

## Dependencies
* Python 3.6
* [PyTorch](http://pytorch.org/), tested with 1.3.1
* [NumPy](http://www.numpy.org/), tested with 1.18.4
* [Pyemd](https://github.com/wmayner/pyemd), fast earth mover distance, tested with 0.5.1
* [Transformers](https://github.com/huggingface/transformers), tested with 2.7.0
* [Mosestokenizer] tokenization from the Moses encoder, tested with 1.0.0

## Usage

### Download re-aligned models and datasets
The models are in the [link](https://drive.google.com/drive/folders/12eQI0-6NbQ9Z3WcaT32WpdWacQAuJwcG?usp=sharing). It is tricky to use shell command to deal with the datasets in Google Drive. So, please kindly download them mannually and move them to the 'model' folder. The RFEval datasets are in the repo already. Please download XNLI datasets in the [link](https://cims.nyu.edu/~sbowman/xnli/) and move to the 'XNLI/dataset' folder.

### Running experiments
For RFEval, we provide experiments for m-BERT and XLM-R and compare the results of three modifications in one go.
For XNLI, you need to fine-tune the models on XNLI so as to complete the experiments. The fine-tuned models are not released in this version. 

```
cd [TASK]
python [MODEL]_eval.py
```

### Ablation results on RFEval 
Modifications                  | m-BERT (fr-en)| XLM-R (fr-en) | m-BERT (de-en) | XLM-R (de-en) |
----------------------- | :------: | :----------: | :----------: | :----------: 
orginal                | 38.2 |  16.8 | 28.3 | 14.8 |
norm_sapce             | 42.4 |  43.7 | 31.8 | 35.5 |
norm_text              | 42.7 |  15.0 | 30.1 | 14.3 |
align                  | 46.1 |  27.6 | 37.0 | 19.3 |
align + norm_space     | 47.6 |  47.6 | 39.4 | **36.9** |
align + norm_space + norm_text  | **52.0** | **47.7** | **40.8** | 36.8 |

We observe that space normalization can achieve the most consistent gains among the three modifications across encoders, tasks and languages. Also, we see the additive effects of the three are often considerable. 

## License

This project is Apache-licensed, as found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## References


```
@article{zhao2020inducing,
  title={Inducing Language-Agnostic Multilingual Representations},
  author={Zhao, Wei and Eger, Steffen and Bjerva, Johannes and Augenstein, Isabelle},
  journal={arXiv preprint arXiv:2008.09112},
  year={2020}
}
```

```
@inproceedings{zhao-etal-2020-limitations,
    title = "On the Limitations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation",
    author = "Zhao, Wei and Glava{\v{s}}, Goran and Peyrard, Maxime and Gao, Yang and West, Robert and Eger, Steffen",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.151",
    pages = "1656--1671"
}
```

