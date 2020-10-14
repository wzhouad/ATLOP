# ATLOP
Code for paper Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling.

If you make use of this code in your work, please kindly cite the following paper:

```bibtex
@article{zhou2020atlop,
  title={Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling},
  author={Zhou, Wenxuan and Huang, Kevin and Ma, Tengyu and Huang, Jing},
  journal={Arxiv},
  year={2020}
}
```

## Requirements
* Python (tested on 3.7)
* [PyTorch](http://pytorch.org/) (tested on 1.4.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 3.3.1)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The expected structure of files is:
```
ATLOP
 |-- dataset
 |    |-- train_annotated.json        
 |    |-- train_distant.json
 |    |-- dev.json
 |    |-- test.json
 |-- meta
 |    |-- rel2id.json
```

## Training and Evaluation
Train the BERT model on DocRED with the following command:

```bash
>> bash ./run_bert.sh  # for BERT
>> bash ./run_roberta.sh  # for RoBERTa
```

The training loss and evaluation scores on the development set will be automatically synced to the wandb dashboard.

The program will generate a test file `result.json` in the official evaluation format. You can compress and submit it to Colab for the official test score.
