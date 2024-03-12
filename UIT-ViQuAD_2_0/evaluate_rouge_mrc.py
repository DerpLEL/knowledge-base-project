from rouge_score import rouge_scorer
import py_vncorenlp
import os
from string import punctuation
import json
from pprint import pprint
import pandas as pd

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
vncorenlp_path = os.path.abspath("../vncorenlp")

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_path)


def preprocess_string(string):
    preprocessed = string.strip().translate(str.maketrans('', '', punctuation)).lower()

    return rdrsegmenter.word_segment(preprocessed)[0]


with open(r'E:\knowledge-base-project\UIT-ViQuAD_2_0\baseline-vimrc-viquad-result.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

results_with_rouge = []

for i in results:
    dct = i
    llm_answer = i['answer']

    if i['ref_answer']:
        ref_answer = i['ref_answer'][0]['text']

        llm_answer_processed = preprocess_string(llm_answer)
        ref_answer_processed = preprocess_string(ref_answer)

        rouge_score = scorer.score(ref_answer_processed, llm_answer_processed)['rouge1']
        dct['rouge1_precision'] = rouge_score.precision
        dct['rouge1_recall'] = rouge_score.recall
        dct['rouge1_f1'] = rouge_score.fmeasure

    else:
        if 'N/A' in llm_answer.strip():
            dct['rouge1_precision'] = 1.0
            dct['rouge1_recall'] = 1.0
            dct['rouge1_f1'] = 1.0

        else:
            dct['rouge1_precision'] = 0.0
            dct['rouge1_recall'] = 0.0
            dct['rouge1_f1'] = 0.0

    results_with_rouge.append(dct)

df = pd.DataFrame(results_with_rouge)
# print(df.columns.tolist())
cols = ['id', 'context', 'question', 'answer', 'ref_answer', 'rouge1_precision', 'rouge1_recall', 'rouge1_f1']
df = df[cols]
df.to_excel(r'E:\knowledge-base-project\UIT-ViQuAD_2_0\baseline-vimrc-viquad.xlsx', index=False)
