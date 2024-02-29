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

with open('possible-questions.json') as f:
    possible_questions = json.load(f)

with open('impossible-questions.json') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

prompt_format = '''[INST] Given context, answer the question. Output "N/A" if no answer can be found.

Context: {context}

Question: {question}
Answer: Here is the most relevant answer from the context: [/INST]'''

result_format = '''Context: {context}

Question: {question}
Answer: {answer}

Reference answer: {ref_answer}'''

qa_result = []

counter = 0
for index, i in enumerate(chosen_set):
    if index == 41 or index == 69:
        continue

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
        counter += 1
        continue

    result_formatted = result_format.format(
        context=context,
        question=question,
        answer=answer,
        ref_answer=str(i['answers'])
    )

    dct = {
        'id': index,
        'text': result_formatted
    }
    qa_result.append(dct)
    print(f'Answered question {index}: {question}')

print(f'### SKIPPED QUESTIONS: {counter} ###')

with open('baseline-urallama-result.json', 'w') as f:
    json.dump(qa_result, f)
