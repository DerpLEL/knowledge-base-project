import pickle

from rdflib import Graph, URIRef, Literal
import time
import os
import re

prefix = True
filename = 'webqsp-kg.pkl' if prefix else 'webqsp-kg-no-prefix.pkl'

# Initialize a new Graph
g = Graph()

if filename not in os.listdir('.'):
    print('Creating new pickle file...')

    t1 = time.perf_counter()

    train_lines = []
    for i in range(1, 5):
        with open(f'train-{i}.txt', 'r', encoding='utf-8') as file:
            train_lines += file.readlines()

    with open('test.txt', 'r', encoding='utf-8') as file:
        test_lines = file.readlines()

    with open('valid.txt', 'r', encoding='utf-8') as file:
        valid_lines = file.readlines()

    lines = train_lines + test_lines + valid_lines

    url = 'http://rdf.freebase.com/ns/'
    # url = ''

    for line in lines:
        # Split the line into components
        parts = line.strip().split('\t')

        # Create URIs for the subject and predicate
        if prefix:
            subject = URIRef(url + parts[0])
            predicate = URIRef(url + parts[1])

            # The object can be a URI or a literal
            # Here we assume it's a literal for simplicity
            object_literal = Literal(parts[2], lang='en')

        else:
            subject = Literal(parts[0])
            predicate = Literal(parts[1])

            # The object can be a URI or a literal
            # Here we assume it's a literal for simplicity
            object_literal = Literal(parts[2], lang='en')

        # Add the triple to the graph
        g.add((subject, predicate, object_literal))

    with open(filename, 'wb') as file:
        pickle.dump(g, file)
        print(f'Dump successfully. Took {time.perf_counter() - t1:.2f}s')

else:
    print('Using pickle...')

    t1 = time.perf_counter()
    with open(filename, 'rb') as file:
        g = pickle.load(file)

    print(f'Loading took {time.perf_counter() - t1:.2f}s')

def query_processor(string: str):
    if 'dateTime' in string:
        return 'skip'

    # new_string = re.sub('PREFIX ns: <http://rdf.freebase.com/ns/>\n', '', string)
    # new_string = re.sub('ns:', '', new_string)

    return re.sub('OR', '||', string)


query = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.072_m3)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.072_m3 ns:government.politician.government_positions_held ?y .\nns:?y ns:government.government_position_held.district_represented ?x .\n}\n"
processed_query = query_processor(query)

if processed_query != 'skip':
    t2 = time.perf_counter()
    qres = g.query(processed_query)
    for i in qres:
        print(i)

    print(f'Querying took {time.perf_counter() - t2:.2f}s')

else:
    print('Query contains datetime, skipping...')
