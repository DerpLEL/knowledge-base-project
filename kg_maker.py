import hashlib
import random
import json

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

prompt_format = '''Given context, create a knowledge graph as a list of Entity (properties:) [relation] Entity (properties:).
Keep in mind only 1 [relation] is allowed in between 2 entities.

Example
Context: Alice is 25 years old and Bob is her friend. Bob lives in Stockholm, which has an area of 188 km2.
Stockholm has a population of 1 million people.

Output:
- Alice (age: 25) [isFriendWith] Bob
- Bob [livesIn] Stockholm (area: 188 km2, population: 1000000)

Context:
{context}

Output:
'''

kg_result = {}

for index, i in enumerate(chosen_set):
    if index == 41 or index == 69:
        continue

    context = i['context']

    if context in kg_result:
        print(f'Duplicate context {index}, skipping')
        continue

    kg_context = ''
    while not kg_context:
        try:
            kg_context = model.generate_content(
                prompt_format.format(
                    context=context
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
            print(f'Bugged context {index}, trying again...')

    kg_result[context] = kg_context
    print(f'Finished processing context {index}: {kg_context}\n')

with open('kg_context.json', 'w') as f:
    json.dump(kg_result, f)
