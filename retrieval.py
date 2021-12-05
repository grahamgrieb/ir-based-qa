import math, string
import spacy
import numpy as np
import json
from heapq import nlargest
import sentence_similarity
from embeddings import Embeddings

class Retrieval:

    def __init__(self, articles_file = 'data/dev_articles.json'):
        self.articles=json.load(open(articles_file, 'rb'))
        self.num_docs=len(self.articles)
        self.word_document_counts={}
    def get_counts(self):
        for article in self.articles:
            article_title=article['document_title']
            article_text=article['document_text']
            article_text=article_text.translate(str.maketrans('', '', string.punctuation)).lower()
            for word in article_text.split():
                if not word in self.word_document_counts:
                    self.word_document_counts[word]={}
                else:
                    if not article_title in self.word_document_counts[word]:
                        self.word_document_counts[word][article_title]=1
                    else:
                        self.word_document_counts[word][article_title]+=1
    def tf_idf(self,word,document):
        if not document in self.word_document_counts[word]:
            return 0
        tf=math.log(self.word_document_counts[word][document],10)+1
        idf=math.log(self.num_docs/len(self.word_document_counts[word]),10)
        return tf*idf
    def score(self,query,document):
        sum=0
        for word in query.split():
            sum+=self.tf_idf(word,document)/self.num_docs
        return sum
    def score_similarity(self,query,document):
        embeddings = Embeddings()
        return sentence_similarity.score_sentence_dataset(embeddings, query,document)
    def top_result(self,query):
        query=query.translate(str.maketrans('', '', string.punctuation)).lower()
        top_score=0
        for article in self.articles:
            article_title=article['document_title']
            a=self.score(query,article_title)
            if a>top_score:
                top_score=a
                result=article_title
        return result
    def top_n_results(self,query,n):
        nlp = spacy.load('en_core_web_sm')
        query=  ' '.join(token.text for token in nlp(query) if token.pos_ in {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'})
        query=query.translate(str.maketrans('', '', string.punctuation)).lower()
        scores={}
        for article in self.articles:
            article_title=article['document_title']
            #scores[article_title]=self.score(query,article_title)
            scores[article_title]=self.score_similarity(query,article_title)
        return nlargest(n, scores, key = scores.get)
        
if __name__ == '__main__':
   a=Retrieval('data/dev_articles.json')
   #a=Retrieval('data/train_articles.json')
   a.get_counts()
   #print(a.top_n_results("Where was John Kerry born?",10))
   print(a.top_n_results("When was Harvard founded?",10))

