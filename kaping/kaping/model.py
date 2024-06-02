"""
This contains a simple script to the pipeline of KAPING
"""
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from kaping.entity_extractor import RefinedEntityExtractor
# from entity_verbalization import RebelEntityVerbalizer
from kaping.entity_injection import MPNetEntityInjector
from werkzeug.utils import secure_filename
import joblib
import json

# mintaka
# cache_path = "E:\\knowledge-base-project\\kaping\\kaping\\mintaka_wikipedia\\"

# webqsp
# cache_path = "E:\\knowledge-base-project\\kaping\\kaping\\webqsp_wikipedia\\"

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
kg = []
with open('translated_kg.txt', 'r', encoding='utf-8') as f:
	raw = f.readlines()
	for i in raw:
		elems = i[1:-2].split(', ')
		kg.append(elems)

# print(kg)
entity_list = list(set([triple[0] for triple in kg]))


def top_k_triple_extractor(question: np.ndarray, triples: np.ndarray, origin: list, k=10, random=False):
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


def create_embeddings():
	emb = model.encode(entity_list)
	return emb


def get_triples(entity_name):
	return [f"({i[0]}, {i[1]}, {i[2]})" for i in kg if i[0] == entity_name]


entity_embedding = create_embeddings()


def custom_verbalizer(entity_title):
	question_embedding = model.encode([entity_title])
	# Get closest entity
	closest_entity = top_k_triple_extractor(question_embedding, entity_embedding, entity_list, 1)
	# print("Closest entity:", closest_entity)

	return get_triples(closest_entity[0])


def pipeline(config, question: str, device=-1):
	"""
	Create a pipeline for KAPING
	:param config: configuration to set up for injector
	:param question: question to apply KAPING
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""

	# define 3 steps
	injector = MPNetEntityInjector(device=device)

	# retrieve entities from given question

	# entity verbalization
	# file_name = f'{secure_filename(question)}.json'
	# if file_name in os.listdir(cache_path):
	# 	with open(os.path.join(cache_path, file_name), 'r', encoding='utf-8') as f:
	# 		knowledge_triples = json.load(f)

	extractor = RefinedEntityExtractor(device=device)
	# verbalizer = RebelEntityVerbalizer(device=device)

	entity_set = extractor(question)

	knowledge_triples = []
	for entity, entity_title in entity_set:
		# knowledge_triples.extend(verbalizer(entity, entity_title))
		triples = custom_verbalizer(entity)
		print("Target triples:", triples)
		knowledge_triples.extend(triples)

	# entity injection as final prompt as input
	prompt = injector([question], knowledge_triples, k=config.k, random=config.random, no_knowledge=config.no_knowledge)

	return prompt
