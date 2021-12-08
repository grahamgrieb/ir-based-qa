import json
if __name__ == '__main__':
    data=json.load(open("data/train-v2.0.json", 'rb'))['data']
    num_with_ans = 0
    num_without_ans = 0
    qa_records = []
    wiki_articles = {}
    
    for article in data:
        
        for i, paragraph in enumerate(article['paragraphs']):
            
            wiki_articles[article['title']+f'_{i}'] = article['title'] + ' ' + paragraph['context']
            
            for questions in paragraph['qas']:
                qa_record = {}
                qa_record['example_id'] = questions['id']
                qa_record['document_title'] = article['title']
                qa_record['question_text'] = questions['question']
                
                try: 
                    qa_record['short_answer'] = questions['answers'][0]['text']
                    num_with_ans += 1
                except:
                    qa_record['short_answer'] = ""
                    num_without_ans += 1
                    
                qa_records.append(qa_record)
        
        
    wiki_articles = [{'document_title':title, 'document_text': text}\
                         for title, text in wiki_articles.items()]
    
    with open('train_articles.json', 'w') as outfile:
        json.dump(wiki_articles, outfile)
    print(f'Data contains {num_with_ans} question/answer pairs with a short answer, and {num_without_ans} without.'+
          f'\nThere are {len(wiki_articles)} unique wikipedia article paragraphs.')

