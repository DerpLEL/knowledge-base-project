"""
This contains a simple script to calculate the matching score after inference
"""

def evaluate(answer: str | list, predicted: str):
	"""
	Evaluate if the predicted answer contains the true answer
	:param answer: true answer (gold label)
	:param predicted: generated answer (predicted label)
	:return: True/False (boolean)
	"""
	if isinstance(answer, str):
		return answer.lower() in predicted.lower()

	for i in answer:
		if i.lower() in predicted.lower():
			return True

	return False


def accuracy(evaluated):
	"""
	Calculate the accuracy
	:param evaluated:
	"""
	return evaluated.count(True) / len(evaluated) 