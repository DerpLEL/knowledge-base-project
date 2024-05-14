"""
This contains a simple script to load the Mintaka dataset in json-file
"""


import json
import random


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
