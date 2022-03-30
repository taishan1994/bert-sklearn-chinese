import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report

from bert_sklearn import BertTokenClassifier
from bert_sklearn import load_model


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_format(filename, idx=3):
    """Read file in CoNLL-2003 shared task format"""
    # read file
    lines = open(filename).read().strip()

    # find sentence-like boundaries
    lines = lines.split("\n\n")

    # split on newlines
    lines = [line.split("\n") for line in lines]

    # get tokens
    tokens = [[l.split()[0] for l in line] for line in lines]

    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]

    # convert to df
    data = {'tokens': tokens, 'labels': labels}
    df = pd.DataFrame(data=data)

    return df


DATADIR = "nlp_data/ner_data/"


def get_data(trainfile=DATADIR + "train.txt",
             devfile=DATADIR + "dev.txt",
             testfile=DATADIR + "test.txt"):
    train = read_format(trainfile, 1)
    print("Train data: %d sentences, %d tokens" % (len(train), len(flatten(train.tokens))))

    dev = read_format(devfile, 1)
    print("Dev data: %d sentences, %d tokens" % (len(dev), len(flatten(dev.tokens))))

    test = read_format(testfile, 1)
    print("Test data: %d sentences, %d tokens" % (len(test), len(flatten(test.tokens))))

    return train, dev, test


train, dev, test = get_data()
X_train, y_train = train.tokens, train.labels
X_dev, y_dev = dev.tokens, dev.labels
X_test, y_test = test.tokens, test.labels

label_list = np.unique(flatten(y_train))
label_list = list(label_list)
print(label_list)

model = BertTokenClassifier(bert_model='model_hub/bert-base-chinese/',
                            max_seq_length=102,
                            label_list=label_list,
                            epochs=3,
                            learning_rate=2e-5,
                            train_batch_size=16,
                            eval_batch_size=32,
                            )
model = model.fit(X_train, y_train)
savefile = 'output/named_entity_recognition.bin'
model.save(savefile)
new_model = load_model(savefile)

f1_dev = new_model.score(X_dev, y_dev)
print("Dev f1: %0.02f"%(f1_dev))

# score model on test data
f1_test = model.score(X_test, y_test)
print("Test f1: %0.02f"%(f1_test))

# get predictions on test data
y_preds = model.predict(X_test)
# print report on classifier stats
print(classification_report(flatten(y_test), flatten(y_preds)))

text = "乔治华盛顿想访问法国。"
tag_predicts  = model.tag_text(text)
