from entity_verbalization import RebelEntityVerbalizer
import json
from entity_extractor import RefinedEntityExtractor

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


def load_dataset():
    """
    Load the Mintaka data set
    :param json_file: path to Mintaka data
    :return: list of Pair (question, answer)
    """

    qa_pairs = []

    with open("mintaka_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    for data in dataset[:10]:
        qa_pairs.append(Pair(question=data['question'],
                              entities=[d['mention'] for d in data['questionEntity']],
                              answer=data['answer'],
                              q_type=data['complexityType']  )
                        )

    return qa_pairs


thingy = RebelEntityVerbalizer(0)
dataset = load_dataset()

for index, i in enumerate(dataset, 0):
    with open(f"mintaka_wikipedia/mintaka_{index}.json", 'w', encoding='utf-8') as f:
        msg = thingy(i.question)
        json.dump(msg, f)
