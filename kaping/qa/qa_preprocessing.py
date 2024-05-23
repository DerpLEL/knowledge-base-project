"""
This contains a simple script to load the Mintaka dataset in json-file
"""


import json
import random
import re

string = "which person directed the movies starred by [John Krasinski]"
match = re.search(r'[(.*?)]', string)

if match:
    result = match.group(1)
    print(result)
else:
    print("No match found")


class Pair:

    def __init__(self, question: str, entities: list, answer: str | list, q_type: str):
        """
        This includes a pair of question-answer in Mintaka dataset
        
        """
        self.question = question
        self.entities = entities
        self.answer = answer
        self.q_type = q_type
        self.pr_answer = None


def load_dataset(json_file: str):
    """
    Load the Mintaka data set
    :param json_file: path to Mintaka data
    :return: list of Pair (question, answer)
    """

    qa_pairs = []

    with open(json_file, encoding="utf-8") as f:
        dataset = json.load(f)

    for data in dataset:
        qa_pairs.append(Pair(question=data['question'], 
                              entities=[d['mention'] for d in data['questionEntity']],
                              answer=data['answer'],
                              q_type=data['complexityType']  )
                        )

    return qa_pairs
    

def load_webqsp():
    qa_pairs = []

    with open("WebQSP.test.json", 'r', encoding='utf-8') as file:
        dataset = json.load(file)['Questions']

    chosen_set = [i for i in dataset if
                  not i['Parses'][0]['Constraints'] and 'dateTime' not in i['Parses'][0]['Sparql']]

    # random.seed(27)
    # chosen_random_questions = random.choices(chosen_set, k=100)

    for data in chosen_set:
        qa_pairs.append(Pair(
            question=data['ProcessedQuestion'],
            entities=[data['Parses'][0]["PotentialTopicEntityMention"]],
            answer=[i["EntityName"] if i["EntityName"] else i["AnswerArgument"] for i in data['Parses'][0]["Answers"]],
            q_type="null"
        ))

    return qa_pairs


def load_metaqa():
    qa_pairs = []

    three_hops = []
    two_hops = []

    with open("metaqa-2-hop.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()

        prev_answer = ""
        prev_line = lines[0]

        for i in lines[1:]:
            if "[" not in i:
                prev_answer += i.strip()

            else:
                question, answers = prev_line.split('\t')
                answers += prev_answer

                answers = answers.strip().split('|')
                res = re.search(r'\[(.*?)]', question)
                if res:
                    entity = [res.group(1)]

                dct = {
                    "question": question.replace("[", '').replace(']', ''),
                    "answers": answers,
                    "entities": entity,
                    "type": '2-hop'
                }

                two_hops.append(dct)
                prev_line = i

    with open("metaqa-3-hop.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()

        prev_answer = ""
        prev_line = lines[0]

        for i in lines[1:]:
            if "[" not in i:
                prev_answer += i.strip()

            else:
                question, answers = prev_line.split('\t')
                answers += prev_answer

                answers = answers.strip().split('|')
                res = re.search(r'\[(.*?)]', question)
                if res:
                    entity = [res.group(1)]

                dct = {
                    "question": question.replace("[", '').replace(']', ''),
                    "answers": answers,
                    "entities": entity,
                    "type": '3-hop'
                }

                three_hops.append(dct)
                prev_line = i

    random.seed(27)
    chosen_set = random.choices(two_hops, k=100) + random.choices(three_hops, k=100)

    for data in chosen_set:
        qa_pairs.append(Pair(
            question=data['question'],
            entities=data['entities'],
            answer=data['answers'],
            q_type=data['type']
        ))

    return qa_pairs
