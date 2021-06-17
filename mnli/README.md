# Fine-tuning for Natural Language Inference (NLI)

Example notebooks using this codebase (on Kaggle):

## Multi-NLI T5 v1.1 fine-tuning

[Fine-tune t5-v1_1-base using Adafactor](https://www.kaggle.com/ceshine/preprocess-and-finetune-t5-1-1-full).

## Multi-lingual NLI dataset

Directly [fine-tune mT5 on the multi-lingual corpus](https://www.kaggle.com/ceshine/pytorch-lightning-finetune-mt5?scriptVersionId=59310458). Accuracy: **0.69778**.

## Transfer learning from English-only corpus

Firstly, [Fine-tune mT5 on English-only corpus (MultiNLI)](https://www.kaggle.com/ceshine/preprocess-and-finetune-mt5?scriptVersionId=57553107).

Then [use the fine-tuned model fo make prediction on the multi-lingual corpus](https://www.kaggle.com/ceshine/mt5-base-mnli-zero-shot?scriptVersionId=53635578). Accuracy: **0.75996**.

Or [fine-tune the model on the multi-lingual corpus again](https://www.kaggle.com/ceshine/pytorch-lightning-finetune-mnli-pretrained-mt5?scriptVersionId=53641701). Accuracy: **0.77555**.
