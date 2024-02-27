import pandas
import json
import re

import pandas as pd

with open('kg_ver-verified.json', encoding='utf-8') as f:
    kg_result = json.load(f)

with open('baseline-verified.json', encoding='utf-8') as f:
    baseline_result = json.load(f)


overall_result = []

for i in range(len(baseline_result)):
    dct = {}

    baseline_content = re.split(r"Context:|Question:|Answer:|Reference answer:", baseline_result[i]['text'])
    kg_content = re.split(r"Context:|Question:|Answer:|Reference answer:", kg_result[i]['text'])

    dct['base_context'] = baseline_content[1]
    dct['kg_context'] = kg_content[1]
    dct['question'] = baseline_content[2]
    dct['base_answer'] = baseline_content[3]
    dct['base_verification'] = baseline_result[i]['sentiment']
    dct['kg_answer'] = kg_content[3]
    dct['kg_verification'] = kg_result[i]['sentiment']
    dct['reference_answer'] = baseline_content[4]

    overall_result.append(dct)

df = pd.DataFrame(data=overall_result)
df.to_excel('base+kg_result.xlsx', index=False)
