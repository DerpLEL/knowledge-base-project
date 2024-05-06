import pickle

from rdflib import Graph, URIRef, Literal
import time
import os
import re

# Initialize a new Graph
g = Graph()

if 'webqsp-kg.pkl' not in os.listdir('.'):
    with open('train.txt', 'r', encoding='utf-8') as file:
        train_lines = file.readlines()

    with open('test.txt', 'r', encoding='utf-8') as file:
        test_lines = file.readlines()

    with open('valid.txt', 'r', encoding='utf-8') as file:
        valid_lines = file.readlines()

    lines = train_lines + test_lines + valid_lines

    # Assuming your text file is named 'triples.txt' and is in a format similar to N-Triples
    with open('triples.txt', 'r') as file:
        url = 'http://rdf.freebase.com/ns/'

        for line in file:
            # Split the line into components
            parts = line.strip().split('\t')

            # Create URIs for the subject and predicate
            subject = URIRef(url + parts[0])
            predicate = URIRef(url + parts[1])

            # The object can be a URI or a literal
            # Here we assume it's a literal for simplicity
            object_literal = Literal(parts[2], lang='en')

            # Add the triple to the graph
            g.add((subject, predicate, object_literal))

    with open('webqsp-kg.pkl', 'wb') as file:
        pickle.dump(g, file)
        print('Dump successfully.')

else:
    print('Using pickle...')

    t1 = time.perf_counter()
    with open('webqsp-kg.pkl', 'rb') as file:
        g = pickle.load(file)

    print(f'Loading took {time.perf_counter() - t1:.2f}s')

def query_processor(string: str):
    if 'dateTime' in string:
        return 'skip'

    return re.sub('OR', '||', string)

query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.03_r3)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.03_r3 ns:location.country.languages_spoken ?x .\n}\n"
processed_query = query_processor(query)

if processed_query != 'skip':
    t2 = time.perf_counter()
    qres = g.query(processed_query)
    for i in qres:
        print(i)

    print(f'Querying took {time.perf_counter() - t2:.2f}s')

else:
    print('Query contains datetime, skipping...')
