from neo4j import GraphDatabase
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from llm.openai import OpenAIChat

GOOGLE_API_KEY = 'AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')
gpt = OpenAIChat('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', temperature=0.7, model_name="gpt-3.5-turbo-0613")

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "jupiter-club-edition-manila-papa-9630")

driver = GraphDatabase.driver(URI, auth=AUTH)

driver.verify_connectivity()


def retriever(query: str):
    prompt = """Given a user query, generate a Neo4j's Cypher database query to get subgraph(s).

Database entities:
Intent (attribute: name:"Rút môn học")
Student (attribute: id)
Course (attribute: name)
Semester (attribute: name)

Database relations:
(Student)-[:CÓ_Ý_ĐỊNH]->(Intent)
(Intent)-[:ĐANG_YÊU_CẦU_RÚT]->(Course)
(Course)-[:TRONG_HỌC_KỲ]->(Semester)

User query: {query}
Cypher query: """

    answer = model.generate_content(
        prompt.format(
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
    print(answer)

    processed_query = '\n'.join(answer.split('\n')[1:-1])

    records, summary, keys = driver.execute_query(
        processed_query,
        database_="neo4j",
    )

    result = []
    for i in records:
        data = i.data()
        if data not in result:
            result.append(data)

    # print('Triples:', triples)

    return result


def llm_context(query: str):
    prompt = '''Given a user query, generate context related to the query from your knowledge.
    
User query: {query}
Generated context: '''

    answer = model.generate_content(
        prompt.format(
            query=query
        ),
        generation_config=genai.types.GenerationConfig(
            temperature=0.5,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    ).text

    return answer


def llm_context_gpt(query: str):
    prompt = '''Given a user query, generate context related to the query from your knowledge.

User query: {query}
Generated context: '''
    # query += ' (Trích từ quy định trường đại học Bách Khoa)'

    answer = gpt.generate(
        [
            {
                'role': 'system',
                'content': "You are a helpful assistant who will generate context related to user questions with your own knowledge."
            },
            {
                'role': 'user',
                'content': f'{query}\nGenerated context: '
            }
        ]
    )

    return answer


def fuse_context(triples, llm_context):
    variable_mapping = {
        's': 'Student',
        'i': 'Intent',
        'c': 'Course',
        'sem': 'Semester'
    }

    relations_left = {
        's': '[:CÓ_Ý_ĐỊNH]',
        'i': '[:ĐANG_YÊU_CẦU_RÚT]',
        'c': '[:TRONG_HỌC_KỲ]'
    }

    relations_right = {
        '[:CÓ_Ý_ĐỊNH]': 'i',
        '[:ĐANG_YÊU_CẦU_RÚT]': 'c',
        '[:TRONG_HỌC_KỲ]': 'sem'
    }

    text = ''
    for index, i in enumerate(triples, 1):
        text += f'Case {index}:\n'
        text += f'''Student {i['s']} -[:CÓ_Ý_ĐỊNH]-> Intent {i['i']}
Intent {i['i']} -[:ĐANG_YÊU_CẦU_RÚT]-> Course {i['c']}
Course {i['c']} -[:TRONG_HỌC_KỲ]-> Semester {i["sem"]}
'''


    template = f'''<General information>
{llm_context}
    
<Specific cases>
{text}'''

    return template


def final_answer(query: str, context: str):
    prompt = '''Given a user query and context, answer the question.

Context:
{context}

User query: {query}
Answer: '''

    answer = model.generate_content(
        prompt.format(
            context=context,
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

    return answer


sample_query = 'Em muốn rút môn Sức bền vật liệu cho học kì 212. Trường hợp này cho rút không ạ?'

# triples = retriever(sample_query)
# print(triples)
# generated_context = llm_context(sample_query)
generated_context = llm_context_gpt(sample_query)
print(generated_context)
# fused = fuse_context(triples, generated_context)
#
# print("Final answer:", final_answer(sample_query, fused))
