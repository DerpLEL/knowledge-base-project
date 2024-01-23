from transformers import BertTokenizer, TFBertModel
from ura import URAAPIGateway
import pandas as pd
from thefuzz import fuzz
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
import os


os.environ["OPENAI_API_KEY"] = os.getenv('openai_key')


llm = ChatOpenAI()

def get_total_distance(similarity_matrix):
    total = 0
    for i in similarity_matrix:
        lst_sum = sum(i)
        total += lst_sum

    return total


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")

# encoded_text = model(tokenizer(text_1, return_tensors='tf')).last_hidden_state
# encoded_text_2 = model(tokenizer(text_2, return_tensors='tf')).last_hidden_state
# encoded_text_3 = model(tokenizer(text_3, return_tensors='tf')).last_hidden_state

# encoded_text = np.array(encoded_text)
# encoded_text_2 = np.array(encoded_text_2)
# print(cosine_similarity(encoded_text[0], encoded_text_2[0]))
# print(cosine_similarity(encoded_text[0], encoded_text_3[0]))


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
# head = ['drugA', 'drugB', 'drugC', 'drugD', 'drugA', 'drugC', 'drugD', 'drugE', 'gene1', 'gene2','gene3', 'gene4', 'gene50', 'gene2', 'gene3', 'gene4']
# relation = ['treats', 'treats', 'treats', 'treats', 'inhibits', 'inhibits', 'inhibits', 'inhibits', 'associated', 'associated', 'associated', 'associated', 'associated', 'interacts', 'interacts', 'interacts']
# tail = ['fever', 'hepatitis', 'bleeding', 'pain', 'gene1', 'gene2', 'gene4', 'gene20', 'obesity', 'heart_attack', 'hepatitis', 'bleeding', 'cancer', 'gene1', 'gene20', 'gene50']
# embeddings = []
#
# for a, b, c in zip(head, relation, tail):
#     print(f"Processing {a} {b} {c}...")
#     embeddings.append(model(tokenizer(f"{a} {b} {c}", return_tensors='tf')).last_hidden_state)
#
# # Create a dataframe
# df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail, 'embeddings': embeddings})
# df.to_pickle("Example_KG.pkl", compression=None)
df: pd.DataFrame = pd.read_pickle("Example_KG.pkl")
# print()
# print(df['embeddings'][0][0][-1])
# print()

ura_llm = URAAPIGateway(
    headers = {"Content-Type": "application/json; charset=utf-8"},
    api_url = 'https://bahnar.dscilab.com:20007/llama/api',
    model_kwargs={"lang": "en", "temprature": 0},
)

G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['head'], row['tail'], label=row['relation'])

query = "I need some painkiller drug, do you have any recommendations?"
query_embedding = model(tokenizer(query, return_tensors='tf')).last_hidden_state
# print(query_embedding[0])

res = [cosine_similarity(query_embedding[0], i[0])[0][0] for i in df['embeddings'].to_list()]
res_idx = list(enumerate(res))
# print(res_idx)
final = [(df[['head', 'relation', 'tail']].iloc[i].to_list(), j) for i, j in res_idx]
final.sort(key=lambda x: x[1], reverse=True)
print("\nTop 2 cosine similarity scoring triples:")
final = final[:2]
print(final)
depths = 2

head_initial_nodes = [i[0][0] for i in final]
tail_initial_nodes = [i[0][2] for i in final]

head_result = []
for i in head_initial_nodes:
    temp = graph_traverse(G, i, [], depths)

    head_result = resolve_duplicates(head_result, temp)

tail_result = []
for i in tail_initial_nodes:
    temp = graph_traverse(G, i, [], depths)

    tail_result = resolve_duplicates(tail_result, temp)

final_results = resolve_duplicates(head_result, tail_result)
# print(f"{final_results = }")

print()
print(triplets_as_string(final_results))

print("Query:", query)
prompt = """Given a collection of Object-Relation-Object, answer the user's question based on facts inferred from the collection.

<Collection>
{document}
<Collection/>

User: {query}
Assistant: """
print("Answer:", llm.predict(prompt.format(document=triplets_as_string(final_results), query=query), stop=['User:', '\n']).strip())
print()
