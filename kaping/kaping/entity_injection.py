"""
This contains the script for a simple off-the-shelf entity injector, framework used here is MPNet

Steps:
	1. Pass all extracted triples and the question into MPNet to turn them into sentence embeddings
	2. Use cosine similarity to find the top-k-triples
	3. Inject all of them together to form the prompt

"""
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random as rand
import json
import requests

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY = '<insert API key here>'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


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


prompt_1 = '''Given a question, generate all entities related to the question.

Question: {query}
Entities: '''

prompt_2 = '''Given a question, generate all relations related to the question.

Question: {query}
Relations: '''

prompt_3 = '''Given a question along with entities and relations, assemble triples (subject, relation, object) for a knowledge graph.

Question: {query}
Entities: {entities}
Relations: {relations}

Triples: '''

def get_background_knowledge_gemsura(string: str):
	entities = get_gemma(prompt_1.format(
		query=string
	))

	relations = get_gemma(prompt_2.format(
		query=string
	))

	triples = get_gemma(prompt_3.format(
		query=string,
		entities=entities,
		relations=relations
	))

	return triples


def get_background_knowledge(query: str):
	entities = model.generate_content(
		prompt_1.format(
			query=query
		),
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

	# print(predicted_answer)

	relations = model.generate_content(
		prompt_2.format(
			query=query
		),
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

	triples = model.generate_content(
		prompt_3.format(
			query=query,
			entities=entities,
			relations=relations,
		),
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

	# print(triples)
	return triples


class MPNetEntityInjector:
	"""This contains the script for a simple off-the-shelf entity injector, framework used here is MPNet

	Steps:
		1: Pass all extracted triples and the question into MPNet to turn them into sentence embeddings
		2: Use cosine similarity to find the top-k-triples
		3: Inject all of them together to form the prompt
	"""
	# basic prompts
	no_knowledge_prompt = "Please answer this question (Short answer, explanations not needed, output N/A if you can't provide an answer)."
	leading_prompt = "Below are facts in the form of the triple meaningful to answer the questions (Short answer, explanations not needed, output N/A if you can't provide an answer)"

	def __init__(self, device=-1):

		# use this model as main model for entity injector
		self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

	def sentence_embedding(self, texts: list[str]):
		"""
		Use MPNET to turn all textual strings into sentence embeddings
		:param texts: list of texts to turn into sentence embeddings
		:return: embedding in form of numpy.ndarray
		"""
		return self.model.encode(texts)

	def top_k_triple_extractor(self, question: np.ndarray, triples: np.ndarray, origin: list, k=10, random=False):
		"""
		Retrieve the top k triples of KGs used as context for the question

		:param question: question in form of sentence embeddings
		:param triples: triples in form of sentence embeddings
		:param origin: original textual triples to index as MPNet can't decode
		:param k: number of triples to retrieve
		:param random: if this is True, retrieve random knowledge 
		:return: list of triples
		"""
		# in case number of triples is fewer than k 
		if len(triples) < k:
			k = len(triples)

		# if random:
		# 	return rand.sample(triples, k)

		if not len(triples.tolist()):
			return []

		# if not the baseline but the top k most similar
		try:
			similarities = cosine_similarity(question, triples)

		except Exception:
			return []

		top_k_indices = np.argsort(similarities[0])[-k:][::-1]

		return [origin[index] for index in top_k_indices]

	def injection(self, question: str, triples=None, no_knowledge=False):
		"""
		Create prompt based on question and retrieved triples

		:param question: question
		:param triples: list of triples (triples are in string)
		:param no_knowledge: if this is True, only add the no_knowledge_prompt only
		:return:
		"""
		# Gemini background knowledge
		try:
			background_kg = get_background_knowledge(question)

		except Exception:
			background_kg = ''

		# Gemsura background knowledge
		# background_kg = get_background_knowledge_gemsura(question)

		triples_as_str = ', '.join(triples)

		# if no_knowledge:
		# 		return f"{MPNetEntityInjector.no_knowledge_prompt} Question: {question} Answer: "
		# else:
		without_background = f"""{MPNetEntityInjector.leading_prompt}
{triples_as_str}

Question: {question}
Answer: """

		with_background = f"""{MPNetEntityInjector.leading_prompt}
{triples_as_str}

{background_kg}

Question: {question}
Answer: """

		return without_background, with_background, triples_as_str, background_kg

	def __call__(self, question: list, triples: list, k=10, random=False, no_knowledge=False, with_background=False):
		"""
		Retrieve the top k triples of KGs used as context for the question

		:param question: 1 question in form [question]
		:param triples: list of triples
		:param k: number of triples to retrieve
		:param random: whether to retrieve random knowledge instead of KAPING
		:param no_knowledge: whether to not use any extra knowledge at all
		:return:
		"""
		assert type(question) == list
		assert type(triples) == list

		if no_knowledge:
			return self.injection(question[0], no_knowledge)

		# use MPNET to turn all into sentence embeddings
		emb_question = self.sentence_embedding(question)
		emb_triples = self.sentence_embedding(triples)
  
		# retrieve the top k triples
		top_k_triples = self.top_k_triple_extractor(emb_question, emb_triples, origin=triples, k=k, random=random)
		
		print("***** Injection *****")
		print(top_k_triples)
		prompt, prompt_background, kaping_triples, background_triples = self.injection(question[0], top_k_triples)
		print(prompt)
		# create prompt as input
		return prompt, prompt_background, kaping_triples, background_triples

