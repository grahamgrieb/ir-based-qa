import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from retrieval import Retrieval
from reader import Reader
import json

passage_predictions_correct=0
answer_predictions={}
readr=Reader()
def predict(args):
    id=args[0]
    question=args[1]
    document=args[2]
    answer_predictions[id]=result = readr.get_answer(question, document)[0]

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
                    predicted_passage_title,predicted_passage_text=retr.top_result(question['question'])
                    if predicted_passage_text[predicted_passage_text.find(' ')+1:] ==  paragraph['context']:
                        passage_predictions_correct+=1
                    questions.append([question['id'],question['question'],predicted_passage_text])
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
        
    with open('predictions_with_retrieval_passages.json', 'w') as outfile:
        json.dump(answer_predictions, outfile)


