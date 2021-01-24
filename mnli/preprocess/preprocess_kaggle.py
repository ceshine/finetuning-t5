import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_train = pd.read_csv("data/kaggle/train.csv")
    train, valid = train_test_split(df_train, test_size=0.1)
    train.to_csv('data/kaggle/train_split.csv', index=False)
    valid.to_csv('data/kaggle/valid.csv', index=False)
