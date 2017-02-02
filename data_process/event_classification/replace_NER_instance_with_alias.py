import re

stock_institutions = ['NASDAQ','Nasdaq','NYSE','New York Stock Exchange','Securities and Exchange Commission','SEC','Renaissance Capital']
definite_patterns = ['(the ([^\s]*? )?company)','(the ([^\s]*? )?firm)']

def replace_ner(sen, ner_tag, choice):
    words = sen.strip().split(' ')
    ners = ner_tag.strip().split(' ')
    if len(ners) != len(words):
        print sen
    else:
        sentence = ''
        last_ner = None
        for i in xrange(len(words)):
            # Process TICKER
            if words[i] == '-LRB-' and i+5 <= len(words):
                ticker = ' '.join(words[i:i+5])
                if re.match('^-LRB- (Nasdaq|NYSE|NASDAQ) : [A-Z]+ -RRB-$',ticker):
                    ners[i+3] = 'TICKER'
                    words[i+3] = '{TICKER}'
            if words[i] == 'symbol':
                for j in xrange(1, 4):
                    if i+j < len(words) and re.match(r'^[A-Z]+$',words[i+j]):
                        ners[i+j] = 'TICKER'
                        words[i+j] = '{TICKER}'
            # mark organization entity
            if ners[i] == 'ORGANIZATION' and last_ner != 'ORGANIZATION':
                sentence += '<org> '
            elif ners[i] != 'ORGANIZATION' and last_ner == 'ORGANIZATION':
                sentence += '</> '
            if ners[i] == 'MONEY' and last_ner != 'MONEY':
                sentence += '<money> '
            elif ners[i] != 'MONEY' and last_ner == 'MONEY':
                sentence += '</> '
            if ners[i] == 'DATE' and last_ner != 'DATE':
                sentence += '<date> '
            elif ners[i] != 'DATE' and last_ner == 'DATE':
                sentence += '</> '
            if ners[i] == 'NUMBER' and last_ner != 'NUMBER':
                sentence += '<number> '
            elif ners[i] != 'NUMBER' and last_ner == 'NUMBER':
                sentence += '</> '
            if ners[i] == 'TIME' and last_ner != 'TIME':
                sentence += '<time> '
            elif ners[i] != 'TIME' and last_ner == 'TIME':
                sentence += '</> '
            if ners[i] == 'PERCENT' and last_ner != 'PERCENT':
                sentence += '<percent> '
            elif ners[i] != 'PERCENT' and last_ner == 'PERCENT':
                sentence += '</> '
            sentence += words[i] + ' '
            last_ner = ners[i]
        # Replace organization with alias
        org_search = re.findall('(<org>.*?</>)', sentence, re.IGNORECASE)
        org_count = 1
        for org in org_search:
            replace = False
            for stock_institution in stock_institutions:
                if stock_institution in org:
                    sentence = sentence.replace(org,'{STOCK_INSTITUTION}')
                    replace = True
                    break
            if not replace:
                if choice == 'A':
                    sentence = sentence.replace(org, '{ORG}')
                elif choice == 'B':
                    sentence = sentence.replace(org, '{ORG_%d}' % org_count)
                else:
                    sentence = sentence.replace(org, '{ORG} %d' % org_count)
                org_count += 1
        money_search = re.findall('(<money>.*?</>)', sentence, re.IGNORECASE)
        for money in money_search:
            sentence = sentence.replace(money,'{MONEY}')

        date_search = re.findall('(<date>.*?</>)', sentence, re.IGNORECASE)
        for date in date_search:
            sentence = sentence.replace(date,'{TIME}')

        number_search = re.findall('(<number>.*?</>)', sentence, re.IGNORECASE)
        for number in number_search:
            sentence = sentence.replace(number,'{NUMBER}')

        time_search = re.findall('(<time>.*?</>)', sentence, re.IGNORECASE)
        for time in time_search:
            sentence = sentence.replace(time,'{TIME}')

        percent_search = re.findall('(<percent>.*?</>)', sentence, re.IGNORECASE)
        for percent in percent_search:
            sentence = sentence.replace(percent,'{NUMBER} percent')

        for pattern in definite_patterns:
            for definite_org in re.findall(pattern,sentence,re.IGNORECASE):
                if choice == 'A':
                    sentence = sentence.replace(definite_org[0], '{ORG}')
                elif choice == 'B':
                    sentence = sentence.replace(definite_org[0], '{ORG_D}')
                else:
                    sentence = sentence.replace(definite_org[0], '{ORG} 1')
                break
        return sentence












