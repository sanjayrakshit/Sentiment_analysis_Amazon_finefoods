import pickle, numpy as np, pandas as pd, re, string
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings('ignore')



class Load_data:
    def __init__(self, batch_size, sequence_length, min_word_freq):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.min_word_freq = min_word_freq
        self.wids = None
        self.raw_data = pd.read_csv('Reviews.csv')
        self.raw_data.dropna(inplace=True)
        self.raw_data.drop_duplicates(subset=['Summary', 'Text', 'Score'], inplace=True)
        # print(self.raw_data.shape)
        self.raw_data = self.raw_data[self.raw_data['Score'] != 3]
        # print(self.raw_data.shape)


    def preprocess(self, x):
        x = re.sub('<[^>]*>', '', x.lower())
        for punc in string.punctuation:
            if "\'" != punc:
                x = x.replace(punc, f" {punc} ")
        x = re.sub(" +", " ", x)
        return x


    def create_ids(self):
        # prepare word ids
        print("Creating ids ....")
        corpus = ""
        for i, j in tqdm(zip(list(self.raw_data['Summary']), list(self.raw_data['Text'])), total=len(self.raw_data)):
            corpus = corpus + self.preprocess(str(i) + " " + str(j) + " ")
        tokens = dict(Counter(corpus.split()))
        for i in list(tokens.keys()):
            if tokens[i] < self.min_word_freq:
                del tokens[i]
        self.wids = {
            item: index+2 for index, item in enumerate(tokens.keys())
        }

    
    def prepare_data(self):
        # prepare data here
        print("Preparing data ....")
        xpart = [self.preprocess(str(i)+" "+str(j)) for i, j in \
        tqdm(zip(list(self.raw_data['Summary']), list(self.raw_data['Text'])), total=len(self.raw_data))]
        xpart = [
            [self.wids.get(word, 1) for word in sen.split()] for sen in tqdm(xpart, total=len(xpart))
        ]
        xpart = pad_sequences(xpart, self.sequence_length)
        ypart  = [np.array([1]) if item>3 else np.array([0]) for item in list(self.raw_data['Score'])]
        self.whole_data = list(zip(xpart, ypart))
        self.train, self.test = train_test_split(self.whole_data, test_size=0.15, random_state=101)




    def get_train_batch(self, i):
        temp = list(zip(*self.train[i*self.batch_size : (i+1)*self.batch_size]))
        return temp[0], temp[1]


    def get_test_batch(self, i):
        temp = list(zip(*self.test[i*self.batch_size : (i+1)*self.batch_size]))
        return temp[0], temp[1]


if __name__ == '__main__':
    l = Load_data(batch_size=128, sequence_length=200, min_word_freq=5)       
    l.create_ids()
    l.prepare_data()
    batch = l.get_train_batch(i=0)
    print(batch)





















