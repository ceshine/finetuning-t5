# T5 Paraphrasing Experiments

## Environment

- Python 3.8.3
- PyTorch >= 1.6
- Transformers >= 3.1.0
- nltk 3.2.5
- PyTorch Lightning 1.1.3
- PyTorch Lightning Spells

## Datasets

- [PAWS: Paraphrase Adversaries from Word Scrambling](https://github.com/google-research-datasets/paws)
- [Quora Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
- [Paraphrase Grouped Corpora](https://www.oxinabox.net/resources/paraphrase_grouped_corpora/)
  - Opinosis
  - MSRP

The datasets are put into the `/data/` folder. Please refer to preprocessing script (for now) for the sub-folder namings of the PAWS dataset.

## Instructions

### Preprocessing Steps

These two scripts will normalize the data and create train/valid/test split:

```bash
python preprocessing/preprocess_quora.py
python preprocessing/preprocess_paws.py
python preprocessing/preprocess_grouped_paraphrases.py data/msrp_distribute/
python preprocessing/preprocess_grouped_paraphrases.py data/opinosis_distribute/ --detokenize
```

Pre-tokenize the dataset:

```bash
python preprocessing/tokenize_dataset.py quora
python preprocessing/tokenize_dataset.py paws
python preprocessing/tokenize_dataset.py msrp
python preprocessing/tokenize_dataset.py opinosis
```

### Training

Example:

```bash
python train_t2t.py --fp16 --epochs 5 --dataset quora+paws+opinosis --batch-size 8 --grad-accu 2 --max-len 64
```

Use `python train_t2t.py --help` to see available options.

## Pre-trained Models

[Demo app on Huggingface Spaces](https://huggingface.co/spaces/ceshine/t5-paraphrasing).

### PAWS and Quora

A pre-trained model trained on both PAWS and Quora datasets [is published on Huggingface Model Hub](https://huggingface.co/ceshine/t5-paraphrase-quora-paws). You can use the model to paraphrase sentences by running the following command:

```bash
python generate.py ceshine/t5-paraphrase-quora-paws --num-outputs 5
```

#### Sample Outputs

![Sample output 1](imgs/sample-output-1.png)

![Sample output 2](imgs/sample-output-2.png)

### PAWS, MSRP, and Opinosis

A pre-trained model trained on both PAWS, MSRP, and Opinosis datasets (the latter two taken from [Paraphrase Grouped Corpora](https://www.oxinabox.net/resources/paraphrase_grouped_corpora/)) [is published on Huggingface Model Hub](https://huggingface.co/ceshine/t5-paraphrase-paws-msrp-opinosis). You can use the model to paraphrase sentences by running the following command:

```bash
python generate.py ceshine/t5-paraphrase-paws-msrp-opinosis --num-outputs 5
```

#### Sample Outputs

Opinosis validation:

![Sample output 3](imgs/sample-output-3.png)

MSRP validation:

![Sample output 4](imgs/sample-output-4.png)
