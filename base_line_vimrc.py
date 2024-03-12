import random
import json

from llm.vimrc import ViMRCLarge

model = ViMRCLarge()

with open('./UIT-ViQuAD 2.0/possible-questions-viquad.json', encoding='utf-8') as f:
    possible_questions = json.load(f)

with open('./UIT-ViQuAD 2.0/impossible-questions-viquad.json', encoding='utf-8') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

prompt_format = '''[INST] Cho một đoạn ngữ cảnh, hãy trả lời câu hỏi. Trả lời ngắn gọn.
Phản hồi "N/A" nếu không có câu trả lời.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: [/INST]'''

qa_result = []

bugged_questions = []
for index, i in enumerate(chosen_set):
    # if index == 41 or index == 69:
    #     continue

    context = i['context']
    question = i['question']

    try:
        answer = model.generate(
            question=question, 
            context=context
        )

        if answer == -1:
            raise Exception('Custom.')

    except Exception:
        print(f'Bugged question {index}, skipping...')
        bugged_questions.append(index)
        continue

    plausible_answer = [] if 'plausible_answers' not in i else i['plausible_answers']

    dct = {
        'id': index,
        'question': question,
        'answer': answer,
        'ref_answer': i['answers'],
        'context': context,
        'plausible_answers': plausible_answer
    }
    qa_result.append(dct)
    print(f'Answered question {index}: {question}')

print(f'### SKIPPED QUESTIONS: {bugged_questions} ###')

with open('baseline-vimrc-viquad-result.json', 'w', encoding='utf-8') as f:
    json.dump(qa_result, f, ensure_ascii=False)
