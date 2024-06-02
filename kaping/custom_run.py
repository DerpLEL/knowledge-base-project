import json
from kaping.model import pipeline
import requests
from pprint import pprint
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class config:
    def __init__(self):
        self.k = 10
        self.no_knowledge = False
        self.random = False


GOOGLE_API_KEY='AIzaSyCD0jcUJYdoAdWLc5Fkb63ZGwMJAksmPbQ'

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


# In 2024, which major
# query = 'Does Ho Chi Minh City University of Technology have any ABET-certified programs?'
# query = 'In 2024, which new majors is Ho Chi Minh City university of technology expecting to enroll new students in?'
query = "In 2024, in which cities/provinces is the competency assessment exam by VNU-HCM held?"
args = config()

_, prompt_background, _, _ = pipeline(args, query, device="0")

# print(prompt_background)
try:
    # predicted_answer = model.generate_content(
    #     prompt_background,
    #     generation_config=genai.types.GenerationConfig(
    #         temperature=0.0,
    #     ),
    #     safety_settings={
    #         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    #     }
    # ).text
    predicted_answer = get_gemma(prompt_background)

except Exception:
    predicted_answer = "Error"

print(prompt_background, end='')
print(predicted_answer)
