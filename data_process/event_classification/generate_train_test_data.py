import random
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path.replace('data_process', 'data')
event = 'ipo' # should be either 'ipo' or 'layoff'
sent_file = open(os.path.join(dir_path, event, 'train_a.txt'),'w')
label_file = open(os.path.join(dir_path, event, 'train_label.txt'),'w')
sent_testfile = open(os.path.join(dir_path, event, 'test_a.txt'),'w')
label_testfile = open(os.path.join(dir_path, event, 'test_label.txt'),'w')
train_nerfile = open(os.path.join(dir_path, event, 'train_ner.txt'),'w')
test_nerfile = open(os.path.join(dir_path, event, 'test_ner.txt'),'w')
label_summary = {}

with open(os.path.join(dir_path, event, 'raw_data/sentence_label.txt'),'r') as f:
    for line in f.readlines():
        sen_id, sen, ner_tag, label, label1 = line.split('~^~')
        if event == 'ipo':
            # This is for IPO label
            if label in ['Upcoming','Delay','Updates','File']:
                label = 'Upcoming'
            elif label in ['N/A','Withdraw','Intention']:
                label = 'N/A'
            else:
                label = 'Priced'
        if label not in label_summary:
            label_summary[label] = []
        label_summary[label].append((sen_id, sen, ner_tag))

train_data = {}
for label in label_summary:
    total_num = len(label_summary[label])
    count = int(total_num*0.1)
    count += 1
    sample_idx = random.sample(range(total_num), count)
    test_data = [label_summary[label][i] for i in xrange(total_num) if i in sample_idx]
    train_data[label] = [label_summary[label][i] for i in xrange(total_num) if i not in sample_idx]
    for sen_id, sen, ner in test_data:
        sent_testfile.write(sen+'\n')
        label_testfile.write(label+'\n')
        test_nerfile.write(ner+'\n')
    print 'Category %s: %d sentences in total,%d for training , %d for testing' % \
          (label, total_num, len(train_data[label]), len(test_data))

sentences = []
for label in train_data:
    for sen in train_data[label]:
        sentences.append((label, sen))

random.shuffle(sentences)
for label, (sen_id, sen, ner) in sentences:
    sent_file.write(sen+'\n')
    label_file.write(label+'\n')
    train_nerfile.write(ner+'\n')






