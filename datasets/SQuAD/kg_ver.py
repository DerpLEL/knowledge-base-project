import random
import json

import pathlib
import textwrap

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY='AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

with open('possible-questions.json') as f:
    possible_questions = json.load(f)

with open('impossible-questions.json') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

with open('kg_context.json') as f:
    kg_context = json.load(f)

prompt_format = '''Given context, answer the question. Output "N/A" if no answer can be found.

Context: {context}

Question: {question}
Answer: Here is the most relevant answer from the context:'''

result_format = '''Context: {context}

Question: {question}
Answer: {answer}

Reference answer: {ref_answer}'''

qa_result = []

counter = 0
for index, i in enumerate(chosen_set):
    if index == 41 or index == 69:
        continue

    context = kg_context[i['context']]
    question = i['question']

    try:
        answer = model.generate_content(
            prompt_format.format(
                context=context,
                question=question
            ),
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        ).text

    except ValueError:
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

with open('kg_ver-result.json', 'w') as f:
    json.dump(qa_result, f)
