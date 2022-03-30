import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report

from bert_sklearn import BertClassifier
from bert_sklearn import load_model


def read_ocnli():
    with open('nlp_data/ocnli_public/train.50k.json', 'r') as fp:
        train_data = fp.readlines()
    train = []
    train_columns = ['text_a', 'text_b', 'label']
    for tdata in train_data:
        try:
            tdata = eval(tdata)
        except Exception as e:
            continue
        train_text_a = tdata['sentence1']
        train_text_b = tdata['sentence2']
        train_label = tdata['label']
        train.append([train_text_a, train_text_b, train_label])
    train_df = pd.DataFrame(train, columns=train_columns)
    with open('nlp_data/ocnli_public/dev.json', 'r') as fp:
        dev_data = fp.readlines()
    dev = []
    dev_columns = ['text_a', 'text_b', 'label']
    for tdata in dev_data:
        tdata = eval(tdata)
        dev_text_a = tdata['sentence1']
        dev_text_b = tdata['sentence2']
        dev_label = tdata['label']
        dev.append([dev_text_a, dev_text_b, dev_label])
    labels = np.unique(train_df['label'])
    dev_df = pd.DataFrame(dev, columns=dev_columns)
    return train_df, dev_df, labels


train, dev, label_list = read_ocnli()
X_train, y_train = train[['text_a', 'text_b']], train['label']
test = dev
X_test, y_test = test[['text_a', 'text_b']], test['label']

model = BertClassifier(bert_model='model_hub/bert-base-chinese/',
                       max_seq_length=128,
                       label_list=label_list,
                       epochs=3,
                       learning_rate=2e-5,
                       train_batch_size=16,
                       eval_batch_size=32,
                       )
model = model.fit(X_train, y_train)
savefile = 'output/text_pair_classification.bin'
model.save(savefile)
new_model = load_model(savefile)
accy = new_model.score(X_test, y_test)
print(accy)

y_pred = new_model.predict(X_test)
print("Accuracy: %0.2f%%"%(metrics.accuracy_score(y_pred, y_test) * 100))
target_names = label_list
print(classification_report(y_test, y_pred, target_names=target_names))
