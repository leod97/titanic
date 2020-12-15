import pandas as pd
import os

train_path = os.path.join(os.path.dirname(__file__),"data", "train.csv")
test_path = os.path.join(os.path.dirname(__file__),"data", "test.csv")

def get_train():
    df = pd.read_csv(train_path)
    return df
    
def get_test():
    df = pd.read_csv(test_path)
    return df

if __name__ == '__main__':
    df = get_train()
    test=get_test()

