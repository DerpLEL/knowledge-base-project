import json

from kaping.model import pipeline
from qa.qa_inference import qa_inference
from qa.qa_evaluate import accuracy, evaluate
from qa.qa_preprocessing import load_dataset, load_webqsp
from arguments import k_parser
import sys
from copy import deepcopy
from langchain_openai import AzureChatOpenAI
import time
import requests

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY='AIzaSyCD0jcUJYdoAdWLc5Fkb63ZGwMJAksmPbQ'

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
	# if not args.input:
	# 	print("No input file, can not run")
	# 	sys.exit(1)

	if args.inference_task == "text2text-generation" and args.model_name == "gpt2":
		print("gpt2 is compatible with text-generation only, change --inference_task if you want to use gpt2")
		sys.exit(1)

	# load mintaka
	# dataset = load_dataset(args.input)

	# load webqsp
	dataset = load_webqsp()

	# set up results
	results = []

	# set up evaluated to calculate the accuracy
	evaluated = []

	results_background = []
	evaluated_background = []

	n = 1

	# ------- run through each question-answer pair and run KAPING
	for index, qa_pair in enumerate(dataset[n-1:], n):
		print(f"{index}. ", args)
		# run KAPING to create prompt
		prompt, prompt_background = pipeline(args, qa_pair.question, device=args.device)

		# No knowledge
# 			prompt = f'''Please answer this question (Short answer, explanations not needed, output N/A if you can't provide an answer).
#
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

		except Exception:
			predicted_answer = "Error"

		# Gemma
		# predicted_answer = get_gemma(prompt)
		print(f'\n### Language model answer: {predicted_answer} ###\n')
		qa_pair.pr_answer = predicted_answer

		with open(f'E:\\knowledge-base-project\\kaping\\gemini-result-question-dump\\kaping_webqsp\\gemini_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer,
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer)
			}
			json.dump(dct, f, ensure_ascii=False)

		# add new qa_pair for output file
		results.append(qa_pair)

		# evaluate webqsp
		evaluated.append(evaluate(qa_pair.answer, predicted_answer))

		# evaluate mintaka
		# evaluated.append(evaluate(qa_pair.answer['mention'], predicted_answer))

		# time.sleep(5)

		# Gemini
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

		with open(f'E:\\knowledge-base-project\\kaping\\gemini-result-question-dump\\kaping_background_webqsp\\gemsura_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt_background,
				"answer": predicted_answer_background,
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer_background)
			}
			json.dump(dct, f, ensure_ascii=False)

		# evaluate webqsp
		evaluated_background.append(evaluate(qa_pair.answer, predicted_answer_background))

		# evaluate mintaka
		# evaluated_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_background))


	msg = ""
	if args.no_knowledge:
		msg = " without knowledge"
	else:
		msg = " with random knowledge" if args.random else " using KAPING"

	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")

	with open('gemmini-webqsp-no-background-knowledge.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated)))

	with open('gemini-webqsp-with-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_background)))

	print(f"Accuracy w/o background: {accuracy(evaluated):2f}")
	print(f"Accuracy with background: {accuracy(evaluated_background):2f}")
	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")


	# output = args.output if args.output else f"./mintaka_predicted_{args.model_name}_{msg[1:]}.csv"
	output = "gemini-webqsp-no-background.csv"
	output_background = "gemini-webqsp-with-background.csv"

	print(f"Save results in {output}")
	with open(output, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# webqsp
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	with open(output_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# webqsp
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
