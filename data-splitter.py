import json

with open('UIT-ViQuAD 2.0/train-v2.0.json', encoding='utf-8') as f:
    dataset = json.load(f)['data']

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
                dct['plausible_answers'] = y['plausible_answers']
                impossible_questions.append(dct)

with open('possible-questions-viquad.json', 'w') as f:
    json.dump(possible_questions, f)

with open('impossible-questions-viquad.json', 'w') as f:
    json.dump(impossible_questions, f)
