import os

from kaping.entity_injection import MPNetEntityInjector
import numpy as np
import json
import ast
# ast.literal_eval("{'muffin' : 'lolz', 'foo' : 'kitty'}")

import re

mintaka = True
use_background = True

folder_path = "gemsura-result-question-dump\\kaping_background_webqsp" if not mintaka else "gemsura-result-question-dump\\kaping_background_mintaka"
injector = MPNetEntityInjector(device=0)

def get_triples(text: str):
    # text = "(Allies, has part, Germany), (World War I, participant, Allies), (Central Powers, conflict, World War I)"

    # Define the regex pattern
    pattern = re.compile(r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)')

    # Find all matches
    matches = pattern.findall(text)

    # Print the matches
    new_matches = [f"({i[0]}, {i[1]}, {i[2]})" for i in matches]
    return new_matches


def parse_answer_webqsp(answer_string: str):
    return answer_string[1:-1].replace("'", '').split(', ')


def parse_answer_mintaka(answer_string: str):
    dct = ast.literal_eval(answer_string)

    result = []
    if 'supportingEnt' in dct:
        for i in dct['supportingEnt']:
            result.append(i['label']['en'])

        return result

    return [dct['mention']]


def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dct = json.load(f)

    return dct


def get_question(prompt: str):
    string = prompt.rsplit("\n\n", 1)[1]
    string = string.replace("Question: ", '').split("\n")[0]

    return string


def get_top_k_triples(question, triples):
    question_emb = injector.sentence_embedding([question])
    triples_emb = injector.sentence_embedding(triples)

    return injector.top_k_triple_extractor(question_emb, triples_emb, origin=triples, k=10, random=False)


def get_hits_at_1(true_answer: str, triples):
    if not mintaka:
        answer_list = parse_answer_webqsp(true_answer)

    else:
        answer_list = parse_answer_mintaka(true_answer)

    for i in answer_list:
        if i.lower() in triples[0].lower():
            return True

    return False


def FindInList(entry, elist):
    for item in elist:
        if str(entry).lower() in str(item).lower() or str(item).lower() in str(entry).lower():
            return True

    return False


def CalculatePRF1(goldAnswerList, predAnswerList):
    if len(goldAnswerList) == 0:
        if len(predAnswerList) == 0:
            return [1.0, 1.0, 1.0]  # consider it 'correct' when there is no labeled answer, and also no predicted answer

        else:
            return [0.0, 1.0, 0.0]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)

    elif len(predAnswerList) == 0:
        return [1.0, 0.0, 0.0]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer

    else:
        # glist = [x["AnswerArgument"] for x in goldAnswerList]
        if not mintaka:
            glist = parse_answer_webqsp(goldAnswerList)

        else:
            glist = parse_answer_mintaka(goldAnswerList)

        plist = []
        for i in predAnswerList:
            entity_1, _, entity_2 = i[1:-1].split(", ")

            if entity_1.lower() not in plist:
                plist.append(entity_1.lower())

            if entity_2.lower() not in plist:
                plist.append(entity_2.lower())

        # plist = predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if FindInList(gentry, plist):
                tp += 1

            else:
                fn += 1

        for pentry in plist:
            if not FindInList(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]


hits_at_1 = 0
failed = 0

f1 = []

files = [i for i in os.listdir(folder_path) if i.endswith("json")]
for index, i in enumerate(files, 1):
    dct = load_file(os.path.join(folder_path, i))
    question = get_question(dct['prompt'])
    answer = dct['actual_answer']
    triples = get_triples(dct['kaping_triples']) + get_triples(dct['background_triples']) if use_background else get_triples(dct['kaping_triples'])

    ranked_triples = get_top_k_triples(question, triples) if use_background else triples
    try:
        hits = get_hits_at_1(answer, ranked_triples)
        if hits:
            hits_at_1 += 1

        f1_entry = CalculatePRF1(answer, ranked_triples)[2]
        f1.append(f1_entry)

    except Exception:
        failed += 1
        print('Bug at question', index, "skipping...")


print(hits_at_1 / len(files))
print("Num failed:", failed)

print(sum(f1) / len(f1))
