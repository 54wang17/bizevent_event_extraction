import random
import os
from collections import Counter
event = 'ipo'
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('data_process', 'data')
sent_file = open(os.path.join(dir_path, event, 'train_a.txt'), 'w')
label_file = open(os.path.join(dir_path, event, 'train_label_a.txt'), 'w')
sent_testfile = open(os.path.join(dir_path, event, 'test_a.txt'), 'w')
label_testfile = open(os.path.join(dir_path, event, 'test_label_a.txt'), 'w')
train_nerfile = open(os.path.join(dir_path, event, 'train_ner_a.txt'), 'w')
test_nerfile = open(os.path.join(dir_path, event, 'test_ner_a.txt'), 'w')

def convertlabel2BIO(labels):
    bio = []
    for idx, label in enumerate(labels):
        if label == 'O':
            bio.append(label)
        elif idx and label == labels[idx-1]:
            bio.append('I-'+label)
        else:
            bio.append('B-'+label)
    return bio



sentences = []
with open(os.path.join(dir_path, event, 'raw_data/sentence_label.txt'), 'r') as f:
    label_summary = Counter()
    data = f.readlines()
    for line in data:
        sen_id, sen, sen_label, ner_tag = line.split('~^~')
        s_labels = []
        for label in sen_label.split(' '):
            label = label.replace('1', '')
            if label.lower() in ["ipo-symbol", "ipo-symbol1", "ipo-price", "ipo-fund", "ipo-fund1", "ipo-share"]:
                label = 'O'
            s_labels.append(label)
        s_labels = convertlabel2BIO(s_labels)
        for label in s_labels:
            label_summary[label] += 1
        sentences.append((sen_id, sen, ' '.join(s_labels), ner_tag))

random.shuffle(sentences)
test_count = int(len(sentences)*0.1)
for idx, (sen_id, sen, sen_label, ner) in enumerate(sentences):
    if idx <= test_count:
        sent_testfile.write(sen+'\n')
        label_testfile.write(sen_label+'\n')
        test_nerfile.write(ner+'\n')
    else:
        sent_file.write(sen+'\n')
        label_file.write(sen_label+'\n')
        train_nerfile.write(ner+'\n')

print label_summary




