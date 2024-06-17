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
import sys
from llm.openai import OpenAIChat
from llm.vimrc import ViMRCLarge

GOOGLE_API_KEY = '<insert API key here>'

genai.configure(api_key=GOOGLE_API_KEY)


model = genai.GenerativeModel('gemini-pro')
# model = OpenAIChat()
# model = ViMRCLarge()


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

	# # some simple tests before running
	# if not args.input:
	# 	print("No input file, can not run")
	# 	sys.exit(1)

	# if args.inference_task == "text2text-generation" and args.model_name == "gpt2":
	# 	print("gpt2 is compatible with text-generation only, change --inference_task if you want to use gpt2")
	# 	sys.exit(1)

	# set up results
	results = []
	results_kaping = []
	results_background = []
	results_kaping_background = []

	# set up evaluated to calculate the accuracy
	evaluated = []
	evaluated_kaping = []
	evaluated_background = []
	evaluated_kaping_background = []
	
	dataset_name = "mintaka"
	model_name = "gemini2"

	# Make directory for model result dump
	import os
	if not os.path.exists(f'{model_name}-result-question-dump'):
		os.makedirs(f'{model_name}-result-question-dump')
	if not os.path.exists(f'{model_name}-result-question-dump\\no_knowledge_{dataset_name}'):
		os.makedirs(f'{model_name}-result-question-dump\\no_knowledge_{dataset_name}')
	if not os.path.exists(f'{model_name}-result-question-dump\\kaping_{dataset_name}'):
		os.makedirs(f'{model_name}-result-question-dump\\kaping_{dataset_name}')
	if not os.path.exists(f'{model_name}-result-question-dump\\background_{dataset_name}'):
		os.makedirs(f'{model_name}-result-question-dump\\background_{dataset_name}')
	if not os.path.exists(f'{model_name}-result-question-dump\\kaping_background_{dataset_name}'):
		os.makedirs(f'{model_name}-result-question-dump\\kaping_background_{dataset_name}')
	
	# Load dataset
	n = 1

	if dataset_name == "mintaka":
		# load mintaka
		dataset = load_dataset("mintaka_dataset.json")

		# Take 1000 questions in Mintaka. Take the first 100 questions and the rest will be randomly selected 900 questions 
		import random

		random.seed(27)
		dataset = dataset[:100] + random.choices(dataset[100:], k=900)

	if dataset_name == "webqsp":
		# load webqsp
		dataset = load_webqsp()

	# ------- run through each question-answer pair and run KAPING
	for index, qa_pair in enumerate(dataset[n-1:], n):
		print(f"{index}. ", args)
		# access the kaping\gemini-result-question-dump\kaping_background_webqsp\gemini_{index}.json
		with open(f'gemini-result-question-dump\\kaping_background_{dataset_name}\\gemini_{index}.json', 'r', encoding='utf-8') as f:
			data = json.load(f)
			# Take in kaping triple and background triple
			kaping_triples = data['kaping_triples']
			background_triples = data['background_triples']

		# # run KAPING to create prompt
		# prompt, prompt_background, kaping_triples, background_triples = pipeline(args, qa_pair.question, device=args.device)

		# No knowledge
		prompt = f'''Answer this question.

		Question: {qa_pair.question}
		Answer: '''
		
		# Kaping knowledge
		prompt_kaping = f'''Answer this question. Use the provided triplets as much as you can.

		Useful information in triplets: {kaping_triples}
		Question: {qa_pair.question}
		Answer: '''

		# Background knowledge
		prompt_background = f'''Answer this question. Use the provided triplets as much as you can.

		Useful information in triplets: {background_triples}
		Question: {qa_pair.question}
		Answer: '''

		# Kaping and background knowledge
		prompt_kaping_background = f'''Answer this question. Use the provided triplets as much as you can.

		Useful information in triplets: {kaping_triples} {background_triples}
		Question: {qa_pair.question}
		Answer: '''

		# # Load the cache result for 2nd run
		# # read no knowledge
		# with open(f'{model_name}-result-question-dump\\no_knowledge_{dataset_name}\\{model_name}_{index}.json', 'r', encoding='utf-8') as f:
		# 	data = json.load(f)
		# 	predicted_answer_no_knowledge = data['answer']
		# # read kaping
		# with open(f'{model_name}-result-question-dump\\kaping_{dataset_name}\\{model_name}_{index}.json', 'r', encoding='utf-8') as f:
		# 	data = json.load(f)
		# 	predicted_answer_kaping = data['answer']
		# # read background
		# with open(f'{model_name}-result-question-dump\\background_{dataset_name}\\{model_name}_{index}.json', 'r', encoding='utf-8') as f:
		# 	data = json.load(f)
		# 	predicted_answer_background = data['answer']
		# # read kaping and background
		# with open(f'{model_name}-result-question-dump\\kaping_background_{dataset_name}\\{model_name}_{index}.json', 'r', encoding='utf-8') as f:
		# 	data = json.load(f)
		# 	predicted_answer_kaping_background = data['answer']
		
		try:
			# Gemini
			# No knowledge
			predicted_answer_no_knowledge = model.generate_content(
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

			# # ViMRC a XLM-Roberta model
			# # No knowledge
			# predicted_answer_no_knowledge = model.generate(qa_pair.question, "")

		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_no_knowledge = "Error"
		
		try:
			# Gemini
			# Kaping knowledge
			predicted_answer_kaping = model.generate_content(
				prompt_kaping,
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

			# # ViMRC a XLM-Roberta model
			# # Kaping knowledge
			# predicted_answer_kaping = model.generate(qa_pair.question, kaping_triples)
			
		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_kaping = "Error"

		try:
			# Gemini
			# Background knowledge
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

			# # ViMRC a XLM-Roberta model
			# # Background knowledge
			# predicted_answer_background = model.generate(qa_pair.question, background_triples)
			
		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_background = "Error"
		
		try:
			# Gemini
			# Kaping and background knowledge
			predicted_answer_kaping_background = model.generate_content(
				prompt_kaping_background,
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

			# # ViMRC a XLM-Roberta model
			# # Kaping and background knowledge
			# predicted_answer_kaping_background = model.generate(qa_pair.question, kaping_triples + background_triples)

		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_kaping_background = "Error"

		# Save the instance result to a file in json format
		print(f'\n### Language model no knowledge answer: {predicted_answer_no_knowledge} ###\n')
		print(f'\n### Language model with kaping answer: {predicted_answer_kaping} ###\n')
		print(f'\n### Language model with background answer: {predicted_answer_background} ###\n')
		print(f'\n### Language model with both answer: {predicted_answer_kaping_background} ###\n')

		# qa_pair.pr_answer = predicted_answer_no_knowledge

		# Save the instance result to a file in json format for each model
		# No knowledge
		# If the result exist in the cache, it won't be overwritten
		if not os.path.exists(f'{model_name}-result-question-dump\\no_knowledge_{dataset_name}\\{model_name}_{index}.json'):
			with open(f'{model_name}-result-question-dump\\no_knowledge_{dataset_name}\\{model_name}_{index}.json', 'w', encoding='utf-8') as f:
				dct = {
					"prompt": prompt,
					"answer": predicted_answer_no_knowledge,
					#"kaping_triples": kaping_triples,
					#"background_triples": background_triples,
					"question": str(qa_pair.question),
					"actual_answer": str(qa_pair.answer),
					"is_correct": evaluate(qa_pair.answer, predicted_answer_no_knowledge) if dataset_name == "webqsp" else evaluate(qa_pair.answer['mention'], predicted_answer_no_knowledge)
				}
				json.dump(dct, f, ensure_ascii=False)

		# Kaping knowledge
		if not os.path.exists(f'{model_name}-result-question-dump\\kaping_{dataset_name}\\{model_name}_{index}.json'):
			with open(f'{model_name}-result-question-dump\\kaping_{dataset_name}\\{model_name}_{index}.json', 'w', encoding='utf-8') as f:
				dct = {
					"prompt": prompt,
					"answer": predicted_answer_kaping,
					"kaping_triples": kaping_triples,
					#"background_triples": background_triples,
					"question": str(qa_pair.question),
					"actual_answer": str(qa_pair.answer),
					"is_correct": evaluate(qa_pair.answer, predicted_answer_kaping) if dataset_name == "webqsp" else evaluate(qa_pair.answer['mention'], predicted_answer_kaping)
				}
				json.dump(dct, f, ensure_ascii=False)
		
		# Background knowledge
		if not os.path.exists(f'{model_name}-result-question-dump\\background_{dataset_name}\\{model_name}_{index}.json'):
			with open(f'{model_name}-result-question-dump\\background_{dataset_name}\\{model_name}_{index}.json', 'w', encoding='utf-8') as f:
				dct = {
					"prompt": prompt,
					"answer": predicted_answer_background,
					#"kaping_triples": kaping_triples,
					"background_triples": background_triples,
					"question": str(qa_pair.question),
					"actual_answer": str(qa_pair.answer),
					"is_correct": evaluate(qa_pair.answer, predicted_answer_background) if dataset_name == "webqsp" else evaluate(qa_pair.answer['mention'], predicted_answer_background)
				}
				json.dump(dct, f, ensure_ascii=False)
		
		# Kaping and background knowledge
		if not os.path.exists(f'{model_name}-result-question-dump\\kaping_background_{dataset_name}\\{model_name}_{index}.json'):
			with open(f'{model_name}-result-question-dump\\kaping_background_{dataset_name}\\{model_name}_{index}.json', 'w', encoding='utf-8') as f:
				dct = {
					"prompt": prompt,
					"answer": predicted_answer_kaping_background,
					"kaping_triples": kaping_triples,
					"background_triples": background_triples,
					"question": str(qa_pair.question),
					"actual_answer": str(qa_pair.answer),
					"is_correct": evaluate(qa_pair.answer, predicted_answer_kaping_background) if dataset_name == "webqsp" else evaluate(qa_pair.answer['mention'], predicted_answer_kaping_background)
				}
				json.dump(dct, f, ensure_ascii=False)
		
		# add new qa_pair for output file
		# no knowledge
		results.append(qa_pair)
		# kaping
		results_kaping.append(qa_pair)
		# background
		results_background.append(qa_pair)
		# kaping and background
		results_kaping_background.append(qa_pair)

		# evaluate webqsp
		if dataset_name == "webqsp":
			evaluated.append(evaluate(qa_pair.answer, predicted_answer_no_knowledge))
			evaluated_kaping.append(evaluate(qa_pair.answer, predicted_answer_kaping))
			evaluated_background.append(evaluate(qa_pair.answer, predicted_answer_background))
			evaluated_kaping_background.append(evaluate(qa_pair.answer, predicted_answer_kaping_background))

		# evaluate mintaka
		else:
			evaluated.append(evaluate(qa_pair.answer['mention'], predicted_answer_no_knowledge))
			evaluated_kaping.append(evaluate(qa_pair.answer['mention'], predicted_answer_kaping))
			evaluated_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_background))
			evaluated_kaping_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_kaping_background))

	# Save accuracy result
	with open(f'{model_name}-{dataset_name}-no-knowledge.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated)))

	with open(f'{model_name}-{dataset_name}-with-kaping.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_kaping)))

	with open(f'{model_name}-{dataset_name}-with-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_background)))

	with open(f'{model_name}-{dataset_name}-with-kaping-with-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_kaping_background)))

	try:
		print(f"Accuracy no knowledge: {str(accuracy(evaluated))}")
		print(f"Accuracy with kaping: {str(accuracy(evaluated_kaping))}")
		print(f"Accuracy with background: {str(accuracy(evaluated_background))}")
		print(f"Accuracy with kaping background: {str(accuracy(evaluated_kaping_background))}")
	except Exception:
		print(f"Error in calculating accuracy w/o background")

	# Define output file
	# no knowledge
	output = f"{model_name}-{dataset_name}-no-knowledge.csv"
	# kaping
	output_kaping = f"{model_name}-{dataset_name}-with-kaping.csv"
	# background
	output_background = f"{model_name}-{dataset_name}-with-background.csv"
	# kaping and background
	output_kaping_background = f"{model_name}-{dataset_name}-with-kaping-with-background.csv"

	# Save result in csv
	# no knowledge
	print(f"Save results in {output}")
	with open(output, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			if dataset_name == "mintaka":
				# mintaka
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			if dataset_name == "webqsp":
				# webqsp
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# kaping
	print(f"Save results in {output_kaping}")
	with open(output_kaping, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			if dataset_name == "mintaka":
				# mintaka
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			if dataset_name == "webqsp":
				# webqsp
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")
	
	# background
	print(f"Save results in {output_background}")
	with open(output_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			if dataset_name == "mintaka":
				# mintaka
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			if dataset_name == "webqsp":
				# webqsp
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# kaping and background
	print(f"Save results in {output_kaping_background}")
	with open(output_kaping_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			if dataset_name == "mintaka":
				# mintaka
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")
				
			if dataset_name == "webqsp":
				# webqsp
				f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
