import json
import os

files = [i for i in os.listdir('.') if i.endswith('json')]

total_list = []
for i in files:
    with open(i, 'r', encoding='utf-8') as f:
        total_list.append(json.load(f))

with open('metaqa-2-and-3-hop.json', 'w', encoding='utf-8') as f:
    json.dump(total_list, f, ensure_ascii=False)
