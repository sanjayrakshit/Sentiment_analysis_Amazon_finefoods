import pandas as pd
import re
import pickle
import warnings

warnings.filterwarnings("ignore")

from Tokenizer import Tokenizer # User defined class

class Load_data:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data

    def load_and_clean(self):
        print ("The columns of the dataframe {}".format(self.data.columns))
        print ("Shape with duplicates : {}".format(self.data.shape))
        # dropping useless colums
        imp_cols = set(self.data.columns)-{"Id","ProductId"}
        print ("important columns : {}".format(imp_cols))
        self.data = self.data.drop_duplicates(subset=imp_cols)
        print("Shape without duplicates : {}".format(self.data.shape))
        # don't want to deal with datapoints when Score = 3
        self.data = self.data[self.data["Score"] != 3]
        # keeping only concerned columns
        Score = self.data["Score"].tolist(); Summary = self.data["Summary"];
        Text = self.data["Text"]
        for loc in range(len(Score)):
            if Score[loc]>3:
                Score[loc] = 1
            else:
                Score[loc] = 0

        return Score, Summary.fillna["Not available"].tolist(), Text.fillna["Not available"].tolist()

    # Utility functions for preprocessing

    def make_lower(self, row):
        return (list(map(lambda x: x.lower(), row)))

    def rem_html(self, row):
        for index in range(len(row)):
            try:
                row[index] = re.sub("<[^>]*>", "", row[index])
            except Exception as e:
                print ("="*30)
                print ("type", type(row[index]))
                print ("Value", row[index])
        return row

    def de_abreviate(self, row):
        with open("converter.DICTIONARY", "rb") as f:
            converter = pickle.load(f)
        for index in range(len(row)):
            for key in converter.keys():
                row[index] = re.sub(key, converter[key], row[index])

        return row

    def create_vocabulary(self, Text, Summary):
        Text = " ".join(Text); Summary = " ".join(Summary)
        ts = Text + " " + Summary
        tobj = Tokenizer(string=ts)
        tokenized_string = tobj.tokenify2()
        tokens = set(tokenized_string.split())
        voc = list(tokens); voc.sort()
        with open("tokens.LIST", "w", encoding="latin") as fp:
            fp.write("\n".join(voc))
        word2ind, ind2word = dict(), dict()
        for index in range(len(voc)):
            word2ind[voc[index]] = index
            ind2word[index] = voc[index]

        return word2ind, ind2word

    def convert2index(self, row, word2index):
        for ind in range(len(row)):
            word_list = row[ind].split()
            for ind2 in range(len(word_list)):
                try:
                    word_list[ind2] = word2index[word_list[ind2]]
                except Exception as e:
                    print ("Exception occured ==== {}".format(e))
                    print ("For word: {}".format(word_list[ind2]))
                    print ("Row: {} ; Column: {}".format(ind, ind2))
            #print (word_list)
            row[ind] = word_list

        return row

    def tokenify_list(self, row):
        for index in range(len(row)):
            row[index] = Tokenizer(string=row[index]).tokenify2()
            row[index] = re.sub("<SOS>","",row[index])
            row[index] = re.sub("<EOS>", "", row[index])
            row[index] = re.sub(" +", " ", row[index])

        return row

    def save_file(self, data, filename, is_binary=False, is_string=False, is_list=False, encoding="utf8"):
        if is_binary:
            with open(filename, "wb") as f:
                pickle.dump(f,data)
        elif is_string:
            with open(filename, "w", encoding=encoding) as f:
                f.write(data)
        elif is_list:
            with open(filename, "w", encoding=encoding) as f:
                f.write("\n".join(data))

if __name__ == "__main__":
    data = pd.read_csv("Reviews.csv")
    data_loader = Load_data(data=data, batch_size=None)
    Score, Summary, Text = data_loader.load_and_clean()

    Summary = Summary.fillna("Not available")
    Text = data_loader.rem_html(row=Text.tolist())
    Summary = data_loader.rem_html(row=Summary.tolist())
    Text = data_loader.make_lower(row=Text)
    Summary = data_loader.make_lower(row=Summary)
    Text = data_loader.de_abreviate(row=Text)
    Summary = data_loader.de_abreviate(row=Summary)
    word2ind, ind2word = data_loader.create_vocabulary(Text=Text, Summary=Summary)
    Text = data_loader.tokenify_list(row=Text)
    Summary = data_loader.tokenify_list(row=Summary)
    Text = data_loader.convert2index(row=Text, word2index=word2ind)
    Summary = data_loader.convert2index(row=Summary, word2index=word2ind)
    print ("Text: {}".format(Text[0]))
    print ("Summary : {}".format(Summary[0]))
    data_loader.save_file(data=(Summary,Text), filename="summarry&text.TUPLE", is_binary=True)
