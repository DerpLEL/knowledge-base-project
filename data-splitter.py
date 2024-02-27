import json

with open('train-v2.0.json') as f:
    dataset = json.load(f)['data']

# 60/40 split, 100 total
possible_questions = []
impossible_questions = []

for i in dataset:
    for x in i['paragraphs']:
        current_paragraph = x['context']

        for y in x['qas']:
            dct = {
                'context': current_paragraph,
                'question': y['question']
            }

            if not y['is_impossible']:
                dct['answers'] = y['answers']
                possible_questions.append(dct)

            else:
                dct['answers'] = []
                impossible_questions.append(dct)

with open('possible-questions.json', 'w') as f:
    json.dump(possible_questions, f)

with open('impossible-questions.json', 'w') as f:
    json.dump(impossible_questions, f)
