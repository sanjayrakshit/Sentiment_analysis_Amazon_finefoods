import re
import string

class Tokenizer:
    
    def __init__(self, file_path = None, encoding = None, string = None):
        if file_path:
            self.file_path = file_path
            with open(self.file_path, "r", encoding=encoding) as fp:
                self.raw_string = fp.read()
        elif string:
            self.raw_string = string
    
    def tokenify1(self):
        raw_string = self.raw_string
        # Spliiting the sentences and adding <EOS> token
        sen_list = raw_string.split(".")
        sen_list.pop(-1)
        sen_list = list(map(lambda x: "<SOS> "+x, sen_list))
        sen_list = list(map(lambda x: x+".<EOS>", sen_list))
        sentences = " ".join(sen_list) + " "
        # Carrying out the tokenization
        tokenized_string = ""
        for ii in range(len(sentences)):
            i = sentences[ii]
            asc = ord(i)
            if(i==" "):
                tokenized_string = tokenized_string + " "
            elif((asc>=65 and asc<=90) or (asc>=97 and asc<=122)):
                tokenized_string = tokenized_string + i
            elif(asc>=48 and asc<=57):
                tokenized_string = tokenized_string + i
            else:
                if(sentences[ii+1] != " "):
                    tokenized_string = tokenized_string + " " + i + " "
                else:
                    tokenized_string = tokenized_string + " " + i
        # rectifying the mistake of tokenizing withing <EOS>
        tokenized_string = re.sub(" < EOS >","<EOS>", tokenized_string)
        tokenized_string = re.sub(" < SOS >","<SOS>", tokenized_string)

        return re.sub(" +", " ", tokenized_string)

    def tokenify2(self):
        raw_string = self.raw_string
        sen_list = raw_string.split(".")
        sen_list.pop(-1)
        sen_list = list(map(lambda x: "<SOS> " + x, sen_list))
        sen_list = list(map(lambda x: x + ".<EOS>", sen_list))
        sentences = " ".join(sen_list)
        for p in string.punctuation:
            sentences = sentences.replace(p, " "+p+" ")
        sentences = re.sub(" < EOS >","<EOS>", sentences)
        sentences = re.sub(" < SOS >","<SOS>", sentences)

        return re.sub(" +", " ", sentences)

    def save_file(self, tokenized_string, encoding):
        file_path = self.file_path
        # Creating the output filename and directory
        file_path = file_path.split("\\")
        revised_output_subpath = "\\".join(file_path[:-1]) + "\\"
        output_file_name = file_path[-1][:-4]+"_tokenized.txt"
        
        #Writing the output to a file
        with open(revised_output_subpath + output_file_name, "w", encoding=encoding) as fp:
            fp.write(tokenized_string)
        fp.close()
        
        
if __name__ == "__main__":
    t = Tokenizer(string="Pharmacotherapeutic group: Vitamin D and analogues,\
     ATC code A11CC03. Alfacalcidol is converted rapidly in the liver to 1,25 \
     dihydroxyvitamin D. This is the metabolite of vitamin D which acts as a\
      regulator of calcium and phosphate metabolism. Since this conversion is\
       rapid, the clinical effects of One-Alpha and 1,25 dihydroxyvitamin D \
       are very similar. Impaired 1 α-hydroxylation by the kidneys reduces \
       endogenous 1,25 dihydroxyvitamin D production. This contributes to the \
       disturbances in mineral metabolism found in several disorders, \
       including renal bone disease, hypoparathyroidism, neonatal hypocalcaemia \
       and vitamin D dependent rickets. These disorders, which require high doses \
       of parent vitamin D for their correction, will respond to small \
       doses of One-Alpha. The delay in response and high dosage required in\
        treating these disorders with parent vitamin D makes dosage adjustment \
        difficult. This can result in unpredictable hypercalcaemia which may \
        take weeks or months to reverse. The major advantage of One-Alpha is\
         the more rapid onset of response, which allows a more accurate \
         titration of dosage. Should inadvertent hypercalcaemia occur it \
         can be reversed within days of stopping treatment.")

    # print (t.tokenify1())
    print (t.tokenify2())
        
        