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
	accuracy_value = (evaluated.count(True) / len(evaluated)) * 100
	return f"{accuracy_value:.2f}%"

# Example usage
test_list = [True, True, False, True]
accuracy_result = accuracy(test_list)
print(accuracy_result)