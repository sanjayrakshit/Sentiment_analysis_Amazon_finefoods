import re
import string

class Tokenizer:
    def __init__(self, file_path=None, encoding=None, string=None):
        if file_path:
            self.file_path = file_path
            with open(self.file_path, "r", encoding="utf8") as f:
                self.raw_string = f.read()
        elif string:
            self.raw_string = string

    def tokenify1(self):
        sen_list = self.raw_string.split(".")
        if sen_list[-1] == "":
            sen_list.pop(-1)
        sen_list = list(map(lambda x: "<SOS>"+x, sen_list))
        sen_list = list(map(lambda x: x+".<EOS>", sen_list))
        sentences = " ".join(sen_list) + " "

        tokenized_string = ""
        for ii in range(len(sentences)):
            i = sentences[ii]
            asc = ord(i)
            if i == " ":
                tokenized_string = tokenized_string + " "
            elif asc>=65 and asc<=95 or asc>=97 and asc<=122:
                tokenized_string = tokenized_string + i
            elif asc>=48 and asc<=57:
                tokenized_string = tokenized_string + i
            else:
                if sentences[ii+1] != " ":
                    tokenized_string = tokenized_string + " " + i + " "
                else:
                    tokenized_string = tokenized_string + " " + i

        tokenized_string = re.sub(" < EOS > ", "<EOS>", tokenized_string)
        tokenized_string = re.sub(" < SOS > ", "<SOS>", tokenized_string)

        return  re.sub(" +"," ",tokenized_string)

    def tokenify2(self):
        sen_list = self.raw_string.split(".")
        if sen_list[-1] == "":
            sen_list.pop(-1)
        sen_list = list(map(lambda x: "<SOS>" + x, sen_list))
        sen_list = list(map(lambda x: x + ".<EOS>", sen_list))
        sentences = " ".join(sen_list)
        for p in string.punctuation:
            sentences = sentences.replace(p, " "+p+" ")
        sentences = re.sub(" < EOS > ", "<EOS>", sentences)
        sentences = re.sub(" < SOS > ", "<SOS>", sentences)

        return re.sub(" +", " ", sentences)