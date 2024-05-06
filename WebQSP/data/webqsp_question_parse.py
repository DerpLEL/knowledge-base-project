import json
from pprint import pprint

with open("WebQSP.test.json", 'r', encoding='utf-8') as file:
    dataset = json.load(file)['Questions']
    print('Original dataset:', len(dataset))

# Exclude all questions with constraints + dateTime reference
# 1215 out of 1639
chosen_set = [i for i in dataset if not i['Parses'][0]['Constraints'] and 'dateTime' not in i['Parses'][0]['Sparql']]

one_hop = []
two_hops = []

for i in chosen_set:
    if i['Parses'][0]['InferentialChain'] and len(i['Parses'][0]['InferentialChain']) > 1:
        two_hops.append(i)

    else:
        one_hop.append(i)

pprint(two_hops[0])
