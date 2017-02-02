import os
from replace_NER_instance_with_alias import replace_ner

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = dir_path.replace('data_process', 'data')
    event = 'ipo'
    print dir_path
    sent_file = open(os.path.join(dir_path, event, 'train_a.txt'), 'r')
    train_ner = open(os.path.join(dir_path, event, 'train_ner.txt'), 'r')
    sent_testfile = open(os.path.join(dir_path, event, 'test_a.txt'), 'r')
    test_ner = open(os.path.join(dir_path, event, 'test_ner.txt'), 'r')

    valid = False
    while not valid:
        choice = raw_input('Please choose which data representation you would like to generate:\n\
                    A. ignore multiple companies\n\
                    B. Use one token to represent different companies\n\
                    C. Use two tokens to represent different companies\n\
                    You are expected to enter letter A or B or C only.\n')
        if choice.upper() not in 'ABC' or len(choice) != 1:
            print 'Invaid input!'
        else:
            valid = True
            choice = choice.upper()
    data_representation_map = {'A': 'b', 'B': 'b1', 'C': 'b2'}
    data_repr = data_representation_map[choice]
    sent_testfile1 = open(os.path.join(dir_path, event, 'test_%s.txt' % data_repr), 'w')
    sent_file1 = open(os.path.join(dir_path, event, 'train_%s.txt' % data_repr), 'w')

    sentences = sent_file.readlines()
    ner_tags = train_ner.readlines()
    assert len(ner_tags) == len(sentences)
    for i in xrange(len(sentences)):
        sentence = replace_ner(sentences[i], ner_tags[i], choice)
        sent_file1.write(sentence+'\n')

    sentences = sent_testfile.readlines()
    ner_tags = test_ner.readlines()
    assert len(ner_tags) == len(sentences)
    for i in xrange(len(sentences)):
        sentence = replace_ner(sentences[i], ner_tags[i], choice)
        sent_testfile1.write(sentence+'\n')