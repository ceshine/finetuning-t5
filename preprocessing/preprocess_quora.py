import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    filename = "data/quora_duplicate_questions.tsv"

    question_pairs = pd.read_csv(filename, sep='\t')
    question_pairs.drop(['qid1', 'qid2', 'id'], axis=1, inplace=True)
    question_pairs_correct_paraphrased = question_pairs[question_pairs['is_duplicate'] == 1].copy()
    question_pairs_correct_paraphrased.drop(['is_duplicate'], axis=1, inplace=True)
    train, test = train_test_split(question_pairs_correct_paraphrased, test_size=0.2)
    valid, test = train_test_split(test, test_size=0.5)
    train.to_csv('data/quora_train.csv', index=False)
    valid.to_csv('data/quora_valid.csv', index=False)
    test.to_csv('data/quora_test.csv', index=False)
