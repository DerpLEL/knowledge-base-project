import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from config import DefaultConfig
import requests


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


CONFIG = DefaultConfig()

llm_base = AzureChatOpenAI(
    openai_api_type="azure",
    azure_endpoint=CONFIG.OPENAI_API_BASE,
    openai_api_version="2023-05-15",
    deployment_name="gpt-35-turbo",
    openai_api_key=CONFIG.OPENAI_API_KEY,
    temperature=0.0,
)

system_1 = """You are an intelligent assistant who analyses questions and outputs related entities.
Output all entities related to the user questions along with other information which may be related.

Question: {query}
Entities: """

GOOGLE_API_KEY='AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

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


print(get_gemma(prompt_2.format(
    query='where george lopez was born?'
)))