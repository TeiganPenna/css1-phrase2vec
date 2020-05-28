import pandas as pd
import re
from nltk.tokenize import word_tokenize
from string import punctuation

df = pd.read_excel('3.Filtered_word_vectors.xlsx', header=0, index_col=0)
Word_List = list(df.index)
Word_Dic ={}

i = 0
for word in Word_List:
    Word_Dic[word] = i
    i += 1
print(len(Word_List))

Term1 = [line.strip('\n') for line in open('Train_Set_Entry_Terms.txt', encoding='UTF-8').readlines()]
Term2 = [line.strip('\n') for line in open('Train_Set_Entry_Terms.txt', encoding='UTF-8').readlines()]
Term_Pair = [(re.sub(r"[{}]+".format(punctuation), '', Term1[i].lower()), re.sub(r"[{}]+".format(punctuation), '', Term2[i].lower())) for i in range(len(Term1))]
Big_Weight = []
index = 1
f = open('Training Set.txt', 'w+', encoding='UTF-8')
for pair in Term_Pair:
    print(index)
    index += 1
    for dimension in range(100):
        weight = [0] * len(Word_List)
        for word in word_tokenize(pair[0]):
            weight[Word_Dic[word]] += df.loc[word][dimension]
        for word in word_tokenize(pair[1]):
            weight[Word_Dic[word]] -= df.loc[word][dimension]
        for element in weight:
            f.write('{}\t'.format(element))
        f.write('\n')
f.close()
