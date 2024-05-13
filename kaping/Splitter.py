import json

with open('mintaka_dataset.json', 'r', encoding='utf-8') as f:
    full_dataset = json.load(f)

ten_sentences = full_dataset[:10]

with open('small_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(ten_sentences, f, ensure_ascii=False)
