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

# model = genai.GenerativeModel('gemini-pro')
# model = OpenAIChat()
model = ViMRCLarge()


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

	# if args.inference_task == "text2text-generation" and args.model_name == "gpt2":
	# 	print("gpt2 is compatible with text-generation only, change --inference_task if you want to use gpt2")
	# 	sys.exit(1)

	# load mintaka
	dataset = load_dataset(args.input)

	# # load webqsp
	# dataset = load_webqsp()

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
	
	# Take 1000 questions in Mintaka. Take the first 100 questions and the rest will be randomly selected 900 questions 
	import random
	n = 1

	random.seed(27)
	dataset = dataset[:100] + random.choices(dataset[100:], k=900)

	# ------- run through each question-answer pair and run KAPING
	for index, qa_pair in enumerate(dataset[n-1:], n):
		print(f"{index}. ", args)
		# access the kaping\gemini-result-question-dump\kaping_background_webqsp\gemini_{index}.json
		with open(f'gemini-result-question-dump\\kaping_background_mintaka\\gemini_{index}.json', 'r', encoding='utf-8') as f:
			data = json.load(f)
			# Take in kaping triple and background triple
			kaping_triples = data['kaping_triples']
			background_triples = data['background_triples']

		# # run KAPING to create prompt
		# prompt, prompt_background, kaping_triples, background_triples = pipeline(args, qa_pair.question, device=args.device)

		# No knowledge
		prompt = f'''Please answer this question (Short answer, explanations not needed, output N/A if you can't provide an answer).
Useful triplet: {kaping_triples + background_triples}
Question: {qa_pair.question}
Answer: '''

		# try:
			# # Gemini
			# predicted_answer = model.generate_content(
			# 	prompt,
			# 	generation_config=genai.types.GenerationConfig(
			# 		temperature=0.0,
			# 	),
			# 	safety_settings={
			# 		HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
			# 		HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
			# 		HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
			# 		HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
			# 	}
			# ).text

			# # GemSURA
			# predicted_answer = get_gemma(prompt)

			# # ChatGPT aka OpenAIChat
			# predicted_answer = model.generate(messages = [prompt])
			# print()

		# except Exception as e:
		# 	print(f"Error: {e}")
		# 	predicted_answer = "Error"

		try:
			# ViMRC a XLM-Roberta model
			# No knowledge
			predicted_answer_no_knowledge = model.generate(qa_pair.question, "")

		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_no_knowledge = "Error"
		
		try:
			# ViMRC a XLM-Roberta model
			# Kaping knowledge
			predicted_answer_kaping = model.generate(qa_pair.question, kaping_triples)
			
		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_kaping = "Error"

		try:
			# ViMRC a XLM-Roberta model
			# Background knowledge
			predicted_answer_background = model.generate(qa_pair.question, background_triples)
			
		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_background = "Error"
		
		try:
			# ViMRC a XLM-Roberta model
			# Kaping and background knowledge
			predicted_answer_kaping_background = model.generate(qa_pair.question, kaping_triples + background_triples)

		except Exception as e:
			print(f"Error: {e}")
			predicted_answer_kaping_background = "Error"

		# Save the instance result to a file in json format
		print(f'\n### Language model answer: {predicted_answer_no_knowledge} ###\n')
		print(f'\n### Language model answer: {predicted_answer_kaping} ###\n')
		print(f'\n### Language model answer: {predicted_answer_background} ###\n')
		print(f'\n### Language model answer: {predicted_answer_kaping_background} ###\n')

		# qa_pair.pr_answer = predicted_answer_no_knowledge

		# Save the instance result to a file in json format for each model
		# No knowledge
		with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\vimrc-result-question-dump\\no_knowledge_mintaka\\vimrc_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer_no_knowledge,
				#"kaping_triples": kaping_triples,
				#"background_triples": background_triples,
				"question": str(qa_pair.question),
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer_no_knowledge) and evaluate(qa_pair.answer['mention'], predicted_answer_no_knowledge)
			}
			json.dump(dct, f, ensure_ascii=False)

		# Kaping knowledge
		with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\vimrc-result-question-dump\\kaping_mintaka\\vimrc_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer_kaping,
				"kaping_triples": kaping_triples,
				#"background_triples": background_triples,
				"question": str(qa_pair.question),
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer_kaping) and evaluate(qa_pair.answer['mention'], predicted_answer_kaping)
			}
			json.dump(dct, f, ensure_ascii=False)
		
		# Background knowledge
		with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\vimrc-result-question-dump\\background_mintaka\\vimrc_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer_background,
				#"kaping_triples": kaping_triples,
				"background_triples": background_triples,
				"question": str(qa_pair.question),
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer_background) and evaluate(qa_pair.answer['mention'], predicted_answer_background)
			}
			json.dump(dct, f, ensure_ascii=False)
		
		# Kaping and background knowledge
		with open(f'D:\\KBQA-project\\knowledge-base-project\\kaping\\vimrc-result-question-dump\\kaping_background_mintaka\\vimrc_{index}.json', 'w', encoding='utf-8') as f:
			dct = {
				"prompt": prompt,
				"answer": predicted_answer_kaping_background,
				"kaping_triples": kaping_triples,
				"background_triples": background_triples,
				"question": str(qa_pair.question),
				"actual_answer": str(qa_pair.answer),
				"is_correct": evaluate(qa_pair.answer, predicted_answer_kaping_background) and evaluate(qa_pair.answer['mention'], predicted_answer_kaping_background)
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

		# # evaluate webqsp
		# evaluated.append(evaluate(qa_pair.answer, predicted_answer))

		# evaluate mintaka
		evaluated.append(evaluate(qa_pair.answer['mention'], predicted_answer_no_knowledge))
		evaluated_kaping.append(evaluate(qa_pair.answer['mention'], predicted_answer_kaping))
		evaluated_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_background))
		evaluated_kaping_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_kaping_background))

		# print(f'\n### Language model answer: {predicted_answer_background} ###\n')
		# qa_pair_copy = deepcopy(qa_pair)

		# qa_pair_copy.pr_answer = predicted_answer_background

		# # add new qa_pair for output file
		# results_background.append(qa_pair_copy)

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

	with open('vimrc-mintaka-no-knowledge.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated)))

	with open('vimrc-mintaka-with-kaping.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_kaping)))

	with open('vimrc-mintaka-with-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_background)))

	with open('vimrc-mintaka-with-kaping-with-background.txt', 'w', encoding='utf-8') as f:
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
	output = "vimrc-mintaka-no-knowledge.csv"
	# kaping
	output_kaping = "vimrc-mintaka-with-kaping.csv"
	# background
	output_background = "vimrc-mintaka-with-background.csv"
	# kaping and background
	output_kaping_background = "vimrc-mintaka-with-kaping-with-background.csv"

	# Save result in csv
	# no knowledge
	print(f"Save results in {output}")
	with open(output, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# # webqsp
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# kaping
	print(f"Save results in {output_kaping}")
	with open(output_kaping, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# # webqsp
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")
	
	# background
	with open(output_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# # webqsp
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	# kaping and background
	with open(output_kaping_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			# mintaka
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")

			# # webqsp
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
