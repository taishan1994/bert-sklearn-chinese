import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

from bert_sklearn import BertClassifier
from bert_sklearn import load_model


def read_tnews():
    with open('nlp_data/tnews_public/labels.json', 'r') as fp:
        train_labels_tmp = fp.readlines()
    labels = []
    for i, tlabel in enumerate(train_labels_tmp):
        tlabel = eval(tlabel)
        label = tlabel['label_desc']
        labels.append(label)
    with open('nlp_data/tnews_public/train.json', 'r') as fp:
        train_data = fp.readlines()
    train = []
    train_columns = ['text', 'label']
    for tdata in train_data:
        tdata = eval(tdata)
        train_text = tdata['sentence']
        train_label = tdata['label_desc']
        train.append([train_text, train_label])
    train_df = pd.DataFrame(train, columns=train_columns)
    with open('nlp_data/tnews_public/dev.json', 'r') as fp:
        dev_data = fp.readlines()
    dev = []
    dev_columns = ['text', 'label']
    for tdata in dev_data:
        tdata = eval(tdata)
        dev_text = tdata['sentence']
        dev_label = tdata['label_desc']
        dev.append([dev_text, dev_label])
    dev_df = pd.DataFrame(dev, columns=dev_columns)
    return train_df, dev_df, labels


train, dev, label_list = read_tnews()
X_train, y_train = train['text'], train['label']
test = dev
X_test, y_test = test['text'], test['label']
model = BertClassifier(bert_model='model_hub/bert-base-chinese/',
                       max_seq_length=128,
                       label_list=label_list,
                       epochs=3,
                       learning_rate=2e-5,
                       train_batch_size=16,
                       eval_batch_size=32,
                       )
model = model.fit(X_train, y_train)
savefile = 'output/text_classification.bin'
model.save(savefile)
new_model = load_model(savefile)
accy = new_model.score(X_test, y_test)
print(accy)

y_pred = new_model.predict(X_test)
print("Accuracy: %0.2f%%"%(metrics.accuracy_score(y_pred, y_test) * 100))
target_names = label_list
print(classification_report(y_test, y_pred, target_names=target_names))
