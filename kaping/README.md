To run, first install the necessary requirements: <br>
```cmd
pip install -r requirements.txt
```

This code is modified from a KAPING implementation found [here](https://github.com/jasmine95dn/kaping_prompt_zero-shot). <br>
run.py:

Change load dataset, and uncomment to switch between webqsp/mintaka and gemini/gemma

model.py:
change cache_path between webqsp and gemma by uncommenting

kaping/entity_injection.py
change background_knowledge between get_background_knowledge (gemini) and get_background_knowledge_gemsura
