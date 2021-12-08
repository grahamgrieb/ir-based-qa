import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from retrieval import Retrieval
from reader import Reader
import json
  
# using
geek_method()
import json

passage_predictions_correct=0
answer_predictions={}
readr=Reader()
def predict(args):
    id=args[0]
    question=args[1]
    documents=args[2]
    answer_predictions[id]=result = readr.get_answer_with_multiple(question, documents)

if __name__ == '__main__':
    f = 'data/dev-v2.0.json'
    retr=Retrieval('data/dev_articles.json')

    data=json.load(open(f, 'rb'))['data']
    questions=[]
    i=0
    for article in data:
        for paragraph in article['paragraphs']:
            for question in paragraph['qas']:
                if(not question['is_impossible']):
                    results=retr.top_n_results(question['question'],5)
                    predicted_passage_title1,predicted_passage_text1=results[0]
                    predicted_passage_title2,predicted_passage_text2=results[1]
                    predicted_passage_title3,predicted_passage_text3=results[2]
                    predicted_passage_title4,predicted_passage_text4=results[3]
                    predicted_passage_title4,predicted_passage_text5=results[4]
                    if (predicted_passage_text1[predicted_passage_text1.find(' ')+1:] ==  paragraph['context']) or (predicted_passage_text2[predicted_passage_text2.find(' ')+1:] ==  paragraph['context']) or (predicted_passage_text3[predicted_passage_text3.find(' ')+1:] ==  paragraph['context']) or (predicted_passage_text4[predicted_passage_text4.find(' ')+1:] ==  paragraph['context']) or (predicted_passage_text5[predicted_passage_text5.find(' ')+1:] ==  paragraph['context']):
                        passage_predictions_correct+=1
                    questions.append([question['id'],question['question'],[predicted_passage_text1,predicted_passage_text2,predicted_passage_text3,predicted_passage_text4,predicted_passage_text5]])
                    i+=1
                    if i%100==0:
                        print(i)
                else:
                    answer_predictions[question['id']]=''
    length=len(questions)
    print('done with retrieval')
    print(f'correct: {passage_predictions_correct}/{length}')
    

    for i,arg in enumerate(questions):
        predict(arg)
        
    with open('predictions_with_topn_retrieval_passages.json', 'w') as outfile:
        json.dump(answer_predictions, outfile)


