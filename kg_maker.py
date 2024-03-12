import random
import json

from llm.gemini import Gemini

# LLM model
model = Gemini()

# Take 100 samples
with open('./UIT-ViQuAD 2.0/possible-questions-viquad.json', encoding='utf-8') as f:
    possible_questions = json.load(f)

with open('./UIT-ViQuAD 2.0/impossible-questions-viquad.json', encoding='utf-8') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

# Prompt
extraction_prompt = '''Given context, create a knowledge graph as a list of Entity (properties:) [relation] Entity (properties:).
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

# Triplet Extraction
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
            kg_context = model.generate(
                extraction_prompt.format(
                    context=context
                )
            )

        except ValueError:
            print(f'Bugged context {index}, trying again...')

    kg_result[context] = kg_context
    print(f'Finished processing context {index}: {kg_context}\n')

with open('./UIT-ViQuAD 2.0/kg_context.json', 'w', encoding='utf-8') as f:
    json.dump(kg_result, f, ensure_ascii=False)
