from kaping.model import pipeline
from qa.qa_inference import qa_inference
from qa.qa_evaluate import accuracy, evaluate
from qa.qa_preprocessing import load_dataset, load_webqsp
from arguments import k_parser
import sys
from copy import deepcopy

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY='AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

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

	# load dataset
	dataset = load_dataset(args.input)
	# dataset = load_webqsp()

	# set up results
	results = []

	# set up evaluated to calculate the accuracy
	evaluated = []

	results_background = []
	evaluated_background = []

	# ------- run through each question-answer pair and run KAPING
	for qa_pair in dataset[:100]:
		print(args)
		# run KAPING to create prompt
		prompt, prompt_background = pipeline(args, qa_pair.question, device=args.device)

		# use inference model to generate predicted answer
		# predicted_answer = qa_inference(task=args.inference_task, model_name=args.model_name,
		# 								prompt=prompt, device=args.device)
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
		print(f'\n### Language model answer: {predicted_answer} ###\n')
		qa_pair.pr_answer = predicted_answer

		# add new qa_pair for output file
		results.append(qa_pair)

		# evaluate to calculate the accuracy
		evaluated.append(evaluate(qa_pair.answer['mention'], predicted_answer))
		# evaluated.append(evaluate(qa_pair.answer, predicted_answer))

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
		print(f'\n### Language model answer: {predicted_answer_background} ###\n')
		qa_pair_copy = deepcopy(qa_pair)

		qa_pair_copy.pr_answer = predicted_answer_background

		# add new qa_pair for output file
		results_background.append(qa_pair_copy)

		# evaluate to calculate the accuracy
		evaluated_background.append(evaluate(qa_pair.answer['mention'], predicted_answer_background))
		# evaluated_background.append(evaluate(qa_pair.answer, predicted_answer_background))

	msg = ""
	if args.no_knowledge:
		msg = " without knowledge"
	else:
		msg = " with random knowledge" if args.random else " using KAPING"

	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")

	with open('gemini-mintaka-100-no-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated)))

	with open('gemini-mintaka-100-with-background.txt', 'w', encoding='utf-8') as f:
		f.write(str(accuracy(evaluated_background)))

	print(f"Accuracy w/o background: {accuracy(evaluated):2f}")
	print(f"Accuracy with background: {accuracy(evaluated_background):2f}")
	# print(f"Accuracy for infering QA task on {args.model_name}{msg}: {accuracy(evaluated):2f}")


	# output = args.output if args.output else f"./mintaka_predicted_{args.model_name}_{msg[1:]}.csv"
	output = "gemini-mintaka-100-no-background.csv"
	output_background = "gemini-mintaka-100-with-background.csv"

	print(f"Save results in {output}")
	with open(output, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

	with open(output_background, 'w', encoding='utf-8') as f2w:
		for qa_pair in results:
			f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer['mention']}\nPredicted answer: {qa_pair.pr_answer}\n\n")
			# f2w.write(f"Question: {qa_pair.question}\nAnswer: {qa_pair.answer}\nPredicted answer: {qa_pair.pr_answer}\n\n")

if __name__ == '__main__':
	main()
