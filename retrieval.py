import math, string
import spacy
import numpy as np
import json
from heapq import nlargest


class Retrieval:

    def __init__(self, articles_file = 'data/dev_articles.json'):
        self.articles=json.load(open(articles_file, 'rb'))
        self.num_docs=len(self.articles)
        #word_document_counts[word][document]
        self.word_document_counts={}
        #tfidfs[word][document]
        self.tfidfs={}
        #document magnitudes, d_mags[document]
        self.d_mags={}

        self.tf_idf()
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
    #fills tfidfs[word][document] with all the tf-idf values and d_mags[document] with all the document magnitudes
    def tf_idf(self):
        self.get_counts()
        for word in self.word_document_counts:
            for document in self.word_document_counts[word]:
                tf=math.log(self.word_document_counts[word][document],10)+1
                idf=math.log(self.num_docs/len(self.word_document_counts[word]),10)
                if not document in self.tfidfs:
                    self.tfidfs[document]={}
                    self.d_mags[document]=0
                self.tfidfs[document][word]=tf*idf
                self.d_mags[document]+=(tf*idf)**2
        
    #scores query,document using tf-idf
    def score(self,query,document):
        sum=0
        for word in query.split():
            if word in self.tfidfs[document]:
                sum+=self.tfidfs[document][word]/math.sqrt(self.d_mags[document])
        return sum
    #returns highest scoring passage for query
    def top_result(self,query):
        query=query.translate(str.maketrans('', '', string.punctuation)).lower()
        top_score=0
        for article in self.articles:
            article_title=article['document_title']
            article_text=article['document_text']
            a=self.score(query,article_title)
            if a>top_score:
                top_score=a
                result=(article_title,article_text)
        return result
    #returns n highest scoring passage for query, a little slower than top_result
    def top_n_results(self,query,n):
        nlp = spacy.load('en_core_web_sm')
        query=  ' '.join(token.text for token in nlp(query) if token.pos_ in {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'})
        query=query.translate(str.maketrans('', '', string.punctuation)).lower()
        scores={}
        for article in self.articles:
            article_title=article['document_title']
            article_text=article['document_text']
            scores[(article_title,article_text)]=self.score(query,article_title)
        return nlargest(n, scores, key = scores.get)
        
if __name__ == '__main__':
   a=Retrieval('data/dev_articles.json')
   #a=Retrieval('data/train_articles.json')
   a.get_counts()
   #print(a.top_n_results("Where was John Kerry born?",10))
   print(a.top_n_results("When was Harvard founded?",1))

