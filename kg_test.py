from ura import URAAPIGateway
import pandas as pd
from thefuzz import fuzz
import networkx as nx
import matplotlib.pyplot as plt

def fuzzy_search(target_lst, query):
    results = []

    for i in target_lst:
        results.append((i, fuzz.partial_ratio(query, i)))

    results.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in results[:1]]

def graph_traverse(G: nx.Graph | nx.DiGraph, initial_node, traversed: list, depth=1):
    results = []
    traversed.append(initial_node)

    # Get all neighbors of node, parents included
    neighbors = list(nx.neighbors(G, initial_node))
    predecessors = list(G.predecessors(initial_node))
    neighbors += [i for i in predecessors if i not in neighbors]

    for i in neighbors:
        if i not in traversed:
            # results.append((initial_node, (G.get_edge_data(initial_node, i))['label'], i))
            if i in predecessors:
                results.append((i, (G.get_edge_data(i, initial_node))['label'], initial_node))

            else:
                results.append((initial_node, (G.get_edge_data(initial_node, i))['label'], i))

            if depth - 1 > 0:
                results += graph_traverse(G, i, traversed, depth - 1)

            traversed.append(i)

    return results

def resolve_duplicates(source_lst, new_lst):
    source_lst += [i for i in new_lst if i not in source_lst]

    return source_lst

def triplets_as_string(lst):
    string = ''

    for i in lst:
        string += f'- {i[0]} {i[1]} {i[2]}\n'

    return string
    
# Define the heads, relations, and tails
head = ['drugA', 'drugB', 'drugC', 'drugD', 'drugA', 'drugC', 'drugD', 'drugE', 'gene1', 'gene2','gene3', 'gene4', 'gene50', 'gene2', 'gene3', 'gene4']
relation = ['treats', 'treats', 'treats', 'treats', 'inhibits', 'inhibits', 'inhibits', 'inhibits', 'associated', 'associated', 'associated', 'associated', 'associated', 'interacts', 'interacts', 'interacts']
tail = ['fever', 'hepatitis', 'bleeding', 'pain', 'gene1', 'gene2', 'gene4', 'gene20', 'obesity', 'heart_attack', 'hepatitis', 'bleeding', 'cancer', 'gene1', 'gene20', 'gene50']

# Create a dataframe
df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})
print()
print(df)

ura_llm = URAAPIGateway(
    headers = {"Content-Type": "application/json; charset=utf-8"},
    api_url = 'https://bahnar.dscilab.com:20007/llama/api',
    model_kwargs={"lang": "en", "temprature": 0},
)

G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['head'], row['tail'], label=row['relation'])

query = "I need some painkiller drug, do you have any recommendations?"
depths = 2

head_initial_nodes = fuzzy_search(df['head'].to_list(), query)
# print(f"{head_initial_nodes = }")
head_result = []

for i in head_initial_nodes:
    temp = graph_traverse(G, i, [], depths)

    head_result = resolve_duplicates(head_result, temp)

# print(f"{head_result =}")
tail_initial_nodes = fuzzy_search(df['tail'].to_list(), query)
# print(f"{tail_initial_nodes = }")
tail_result = []

for i in tail_initial_nodes:
    temp = graph_traverse(G, i, [], depths)

    tail_result = resolve_duplicates(tail_result, temp)

# print(f"{tail_result = }")

final_results = resolve_duplicates(head_result, tail_result)
# print(f"{final_results = }")
print()
print(triplets_as_string(final_results))

print("Query:", query)  
prompt = """[INST]
Given a collection of Object-Relation-Object, answer the user's question based on facts inferred from the collection.

<Collection>
{document}
<Collection/>

User: {query}
Assistant: [/INST]"""
print("Answer:", ura_llm(prompt=prompt.format(document=triplets_as_string(final_results), query=query), stop=['User:']).strip())
print()

# Draw graph
# pos = nx.spring_layout(G, seed=42, k=0.9)
# labels = nx.get_edge_attributes(G, 'label')
# plt.figure(figsize=(12, 10))
# nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
# plt.title('Knowledge Graph')
# plt.show()
