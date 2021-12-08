from transformers import BertForQuestionAnswering
from collections import OrderedDict
from retrieval import Retrieval
import torch
import numpy as np
from transformers import BertTokenizer,AutoTokenizer
class Reader:
    def __init__(self):
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        #self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.max_token_size = self.model.config.max_position_embeddings


    def get_answer(self, question, document):
        #inputs = self.tokenizer.encode_plus(question, document, return_tensors="pt") 
        inputs = self.tokenizer.encode_plus(question, document, return_tensors="pt",max_length=self.max_token_size, truncation="only_second", return_overflowing_tokens=True, stride=20,padding=True)
        #if the token is bigger than the encoder can handle then we need to do the sliding window talked about on page 13 of chapter 23
        if len(inputs["input_ids"].tolist()) > 1:
            #print(f'WENT OVER {question}')
            #encodes question and document into one token, if greater than max token size then the function will return multiple tokenized chunks, truncation="only_second" ensures only the passage part is split among the encodings not the question, stride=20 means there is an overlap of 20 tokens between chunks
            max_val=0
            answer=''
            for i in range(len(inputs['input_ids'].tolist())):
                #print(torch.tensor([inputs['input_ids'].tolist()[i]]))
                answer_start, answer_end = self.model(torch.tensor([inputs['input_ids'].tolist()[i]],dtype=torch.int32),return_dict=False)
                val=(torch.max(answer_start)*torch.max(answer_end)).item()
                answer_start = torch.argmax(answer_start)
                answer_end = torch.argmax(answer_end)+1
                ans=self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][answer_start:answer_end]))
                if (val>max_val): #and (not ans == '[SEP]') and (not ans == '[PAD]') and (not ans == '[CLS]') :
                    max_val=val
                    answer=ans
                    
            if  (answer == '[SEP]') or (answer == '[PAD]') or (answer == '[CLS]'):
                print("broke")
        else:
            #run the model
            #answer_start, answer_end = self.model(inputs['input_ids'], return_dict=False)
            answer_start, answer_end = self.model(torch.tensor([inputs['input_ids'].tolist()[0]],dtype=torch.int32), return_dict=False)
            max_val=(torch.max(answer_start)*torch.max(answer_end)).item()
            #choose best start and end index
            answer_start = torch.argmax(answer_start)
            answer_end = torch.argmax(answer_end)+1
            #use start and end token to get span in string form
            answer=self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        return answer, max_val
    #gets answer for each document and chooses the highest scoring one
    def get_answer_with_multiple(self,question,documents):
        max_val=0
        for document in documents:
            answer,val=self.get_answer(question,document)
            if val > max_val:
                max_val=val
                max_ans=answer
        return max_ans



if __name__ == '__main__':
    #question="Who ruined Roussel de Bailleul's plans for an independent state?"
    question="Where was John Kerry born?"
    a=Retrieval('data/train_articles.json')
    top_result_title,top_result_text=a.top_result(question)
    read=Reader()
    answer=read.get_answer(question, top_result_text)
    
    print('Answer: "' + answer + '"')
