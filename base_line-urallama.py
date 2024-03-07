import random
import json
from urallama import URAWrapper

# import google.generativeai as genai
#
# GOOGLE_API_KEY='AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'
#
# genai.configure(api_key=GOOGLE_API_KEY)
#
# model = genai.GenerativeModel('gemini-pro')

model = URAWrapper()

with open('possible-questions-viquad.json', encoding='utf-8') as f:
    possible_questions = json.load(f)

with open('impossible-questions-viquad.json', encoding='utf-8') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

# prompt_format = '''[INST] Given context, answer the question. Output "N/A" if no answer can be found.
#
# Context: {context}
#
# Question: {question}
# Answer: Here is the most relevant answer from the context: [/INST]'''

prompt_format = '''[INST] Cho một đoạn ngữ cảnh, hãy trả lời câu hỏi. Trả lời ngắn gọn.
Phản hồi "N/A" nếu không có câu trả lời.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: [/INST]'''

# result_format = '''Context: {context}
#
# Question: {question}
# Answer: {answer}
#
# Reference answer: {ref_answer}'''

qa_result = []

bugged_questions = []
for index, i in enumerate(chosen_set):
    # if index == 41 or index == 69:
    #     continue

    context = i['context']
    question = i['question']

    try:
        answer = model.predict(
            prompt_format.format(
                context=context,
                question=question
            )
        )

        if answer == -1:
            raise Exception('Custom.')

    except Exception:
        print(f'Bugged question {index}, skipping...')
        bugged_questions.append(index)
        continue

    # result_formatted = result_format.format(
    #     context=context,
    #     question=question,
    #     answer=answer,
    #     ref_answer=str(i['answers'])
    # )

    plausible_answer = [] if 'plausible_answers' not in i else i['plausible_answers']

    dct = {
        'id': index,
        'answer': answer,
        'ref_answer': str(i['answers']),
        'context': context,
        'plausible_answers': plausible_answer
    }
    qa_result.append(dct)
    print(f'Answered question {index}: {question}')

print(f'### SKIPPED QUESTIONS: {bugged_questions} ###')

with open('baseline-urallama-viquad-result.json', 'w', encoding='utf-8') as f:
    json.dump(qa_result, f, ensure_ascii=False)
