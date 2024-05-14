"""
This contains a simple script to the pipeline of KAPING
"""
import os

from kaping.entity_extractor import RefinedEntityExtractor
from kaping.entity_verbalization import RebelEntityVerbalizer
from kaping.entity_injection import MPNetEntityInjector
from werkzeug.utils import secure_filename
import json

# mintaka_path = "E:\\knowledge-base-project\\kaping\\kaping\\mintaka_wikipedia\\"
mintaka_path = "E:\\knowledge-base-project\\kaping\\kaping\\webqsp_wikipedia\\"


def pipeline(config, question: str, device=-1):
	"""
	Create a pipeline for KAPING
	:param config: configuration to set up for injector
	:param question: question to apply KAPING
	:param device: choose the device to run this pipeline on, for upgrading to GPU change this to  (default: -1, which means for CPU)
	:return: final prompt as output of KAPING to feed into a QA model
	"""

	# define 3 steps
	extractor = RefinedEntityExtractor(device=device)
	verbalizer = RebelEntityVerbalizer(device=device)
	injector = MPNetEntityInjector(device=device)

	# retrieve entities from given question

	# entity verbalization
	file_name = f'{secure_filename(question)}.json'
	if file_name in os.listdir(mintaka_path):
		with open(os.path.join(mintaka_path, file_name), 'r', encoding='utf-8') as f:
			knowledge_triples = json.load(f)

	else:
		entity_set = extractor(question)

		knowledge_triples = []
		for entity, entity_title in entity_set:
			knowledge_triples.extend(verbalizer(entity, entity_title))

		with open(os.path.join(mintaka_path, file_name), 'w', encoding='utf-8') as f:
			json.dump(knowledge_triples, f, ensure_ascii=False)
 
	# entity injection as final prompt as input
	prompt = injector([question], knowledge_triples, k=config.k, random=config.random, no_knowledge=config.no_knowledge)

	return prompt

	

