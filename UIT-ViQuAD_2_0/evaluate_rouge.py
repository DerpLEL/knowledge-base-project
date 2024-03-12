from rouge_score import rouge_scorer
import py_vncorenlp
import os
from string import punctuation
import json

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
vncorenlp_path = os.path.abspath("../vncorenlp")

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)


def preprocess_string(string):
    preprocessed = string.strip().translate(str.maketrans('', '', punctuation)).lower()

    return rdrsegmenter.word_segment(preprocessed)[0]


with open('baseline-urallama-viquad-result.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

results_with_rouge = []

for i in results:
    dct = i
    llm_answer = i['answer']
    ref_answer = i['ref_answer']['text']

    llm_answer_processed = preprocess_string(llm_answer)
    ref_answer_processed = preprocess_string(ref_answer)

    dct['rouge_score'] = scorer.score(ref_answer_processed, llm_answer_processed)

    results_with_rouge.append(dct)

print(results_with_rouge)
