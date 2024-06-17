import json

from kaping.model import pipeline
from qa.qa_inference import qa_inference
from qa.qa_evaluate import accuracy, evaluate
from qa.qa_preprocessing import load_dataset, load_webqsp
from arguments import k_parser
import sys
from copy import deepcopy
import time
import requests

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY = '<insert API key here>'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

no_knowledge = False


def get_gemma(string: str):
    x = requests.post(
        'https://ws.gvlab.org/fablab/ura/llama/api/generate',
        headers={
            'Content-Type': 'application/json'
        },
        json={
            "inputs": f"<start_of_turn>user\n{string}<end_of_turn>\n<start_of_turn>model\n",
        }
    )

    return x.json()['generated_text']


def main():
	# load arguments
	args = k_parser()

	# some simple tests before running
	if not args.input:
		print("No input file, can not run")
		sys.exit(1)

	if args.inference_task == "text2text-generation" and args.model_name == "gpt2":
		print("gpt2 is compatible with text-generation only, change --inference_task if you want to use gpt2")
		sys.exit(1)

	# load mintaka
	dataset = load_dataset(args.input)

	# load webqsp
	# dataset = load_webqsp()

	# set up results
	results = []

	# set up evaluated to calculate the accuracy
	evaluated = []

	results_background = []
	evaluated_background = []

	# Take 1000 questions in Mintaka. Take the first 100 questions and the rest will be randomly selected 900 questions 
	import random
	n = 1

	random.seed(27)
	dataset = dataset[:100] + random.choices(dataset[100:], k=900)

	# ------- run through each question-answer pair and run KAPING
	for index, qa_pair in enumerate(dataset[n-1:], n):
		print(f"{index}. ", args)
		# run KAPING to create prompt
		prompt, prompt_background, kaping_triples, background_triples = pipeline(args, qa_pair.question, device=args.device)

		# No knowledge
# 		prompt = f'''Please answer this question (Short answer, explanations not needed, output N/A if you can't provide an answer).

# Question: {qa_pair.question}
# Answer: '''

		try:
			predicted_answer = model.generate_content(
				prompt,
				generation_config=genai.types.GenerationConfig(
					temperature=0.0,
				),
				safety_settings={
					HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
				}
			).text
			#predicted_answer = get_gemma(prompt)

		except Exception:
			predicted_answer = "Error"

		# Gemma
		# predicted_answer = get_gemma(prompt)
		print(f'\n### Language model answer: {predicted_answer} ###\n')
		qa_pair.pr_answer = predicted_answer

		with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\gemsura-result-question-dump\\no_knowledge_mintaka\\gemsura_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer,
				# "kaping_triples": kaping_triples,
				# "background_triples": background_triples,
				"question": str(qa_pair.question),
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer)
			}
			json.dump(dct, f, ensure_ascii=False)

		# add new qa_pair for output file
		results.append(qa_pair)

		# evaluate webqsp
		# evaluated.append(evaluate(qa_pair.answer, predicted_answer))

		# evaluate mintaka
		evaluated.append(evaluate(qa_pair.answer['mention'], predicted_answer))

		# time.sleep(5)

		# Gemini with background knowledge
		try:
			predicted_answer_background = model.generate_content(
				prompt_background,
				generation_config=genai.types.GenerationConfig(
					temperature=0.0,
				),
				safety_settings={
					HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
					HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
				}
			).text

		except Exception:
			predicted_answer_background = 'Error'

		# Gemma
		# predicted_answer_background = get_gemma(prompt_background)

		print(f'\n### Language model answer: {predicted_answer_background} ###\n')
		qa_pair_copy = deepcopy(qa_pair)

		qa_pair_copy.pr_answer = predicted_answer_background

		# add new qa_pair for output file
		results_background.append(qa_pair_copy)

		# with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\gemini-result-question-dump\\kaping_background_webqsp\\gemini_{index}.json', 'w', encoding='utf-8') as f:
		# 	dct = {
		# 		"prompt": prompt_background,
		# 		"answer": predicted_answer_background,
		# 		"kaping_triples": kaping_triples,
		# 		"background_triples": background_triples,
		# 		"actual_answer": str(qa_pair.answer),
		# 		"is_correct": evaluate(qa_pair.answer, predicted_answer_background)
		# 	}
		# 	json.dump(dct, f, ensure_ascii=False)

		# # evaluate webqsp
		# evaluated_background.append(evaluate(qa_pair.answer, predicted_answer_background))

		# evaluate mintaka
		# evaluated_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_background))


	msg = ""
	if args.no_knowledge:
		msg = " without knowledge"
	else:
		msg = " with random knowledge" if args.random else " using KAPING"

	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")

	with open('gemsura-mintaka-no-knowledge.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated)))

	# with open('gemini-webqsp-with-background.txt', 'w', encoding='utf-8') as f:
	# 	f.write(str(accuracy(evaluated_background)))

	try:
		print(f"Accuracy w/o background: {str(accuracy(evaluated))}")
	except Exception:
		print(f"Error in calculating accuracy w/o background")

	# print(f"Accuracy with background: {accuracy(evaluated_background):2f}")
	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")


	# output = args.output if args.output else f"./mintaka_predicted_{args.model_name}_{msg[1:]}.csv"
	# output = "gemini-webqsp-no-background.csv"
	# output_background = "gemini-webqsp-with-background.csv"

	# print(f"Save results in {output}")
	# with open(output, 'w', encoding='utf-8') as f2w:
	# 	for qa_pair in results:
	# 		# mintaka
	# 		# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# 		# webqsp
	# 		f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# with open(output_background, 'w', encoding='utf-8') as f2w:
	# 	for qa_pair in results:
	# 		# mintaka
	# 		# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# 		# webqsp
	# 		f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
