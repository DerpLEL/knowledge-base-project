# Quickstart
This code is modified from a KAPING implementation found [here](https://github.com/jasmine95dn/kaping_prompt_zero-shot). <br>

To run, first install the necessary requirements: <br>
```cmd
pip install -r requirements.txt
```
To run the code, run this command in this folder: <br>
```cmd
python run.py
```

If you have a GPU with CUDA installed, do: <br>
```cmd
python run.py --device 0
```
run.py can be changed with other run files (run-no.py, etc.) included in this folder.

# Structure
```bash
.
├── README.md
├── arguments.py
├── get_hits_f1.py
├── kaping
│   ├── mintaka_wikipedia
│   ├── webqsp_wikipedia
│   ├── create_rebel.py
│   ├── entity_extractor.py
│   ├── entity_injection.py
│   ├── entity_verbalization.py
│   └── model.py
├── llm
│   ├── model
│   ├── __init__.py
│   ├── basellm.py
│   ├── gemini.py
│   ├── openai.py
│   ├── ura.py
│   └── vimrc.py
├── qa
│   ├── qa_evaluate.py
│   ├── qa_inference.py
│   └── qa_preprocessing.py
├── overall-result-dump
│   ├── gemini-1.0-pro
│   ├── gemsura-7b
│   ├── gpt-3.5-turbo
│   └── vimrc
├── result-question-dump
│   ├── gemini-1.0-pro
│   ├── gemsura-7b
│   ├── gpt-3.5-turbo
│   └── vimrc
├── requirements.txt
├── run.py
├── run-no.py
├── run-webqsp-gemini.py
├── run-webqsp-gemsura.py
├── mintaka_dataset.json
└── WebQSP.test.json
```

# Other run files
1. run.py: The base run file, for running the model (Background knowledge + External knowledge) and outputting the result files
2. run-no.py: Modified run file, runs the model with no knowledge
3. run-webqsp-gemini.py: Runs the model using Gemini on WebQSP dataset, both no knowledge and Background knowledge + External knowledge
4. run-webqsp-gemsura.py: like 3. but for GemSUra-7b instead

# Other notes
kaping/model.py:
- Change cache_path between webqsp and gemma by uncommenting, cache is necessary since REBEL is slow when processing wiki pages. 
- You can apply other methods of querying KG and using them as context here and bypass REBEL entirely.

kaping/entity_injection.py:
- Change background_knowledge between get_background_knowledge (gemini) and get_background_knowledge_gemsura, you can add functions for other models if needed.

qa/preprocessing.py:
- You can add more functions to handle more datasets aside from WebQSP and mintaka, and then add the dataset loading functions back into the run files