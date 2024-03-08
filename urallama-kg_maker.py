import hashlib
import random
import json

from urallama import URAWrapper

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

prompt_format = '''[INST] Given context, create a knowledge graph as a list of Entity (properties:) [relation] Entity (properties:).

Context:
Alice is 25 years old and Bob is her friend. Bob lives in Stockholm, which has an area of 188 km2.
Stockholm has a population of 1 million people.

Output: - Alice (age: 25) [isFriendWith] Bob
- Bob [livesIn] Stockholm (area: 188 km2, population: 1000000)

Context:
{context}

Output: Here is a knowledge graph for the given context: [/INST]'''

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
            kg_context = model.predict(
                prompt_format.format(
                    context=context
                ),
            )

        except ValueError:
            print(f'Bugged context {index}, trying again...')

    kg_result[context] = kg_context
    print(f'Finished processing context {index}: {kg_context}\n')

with open('urallama-kg_context.json', 'w') as f:
    json.dump(kg_result, f)
