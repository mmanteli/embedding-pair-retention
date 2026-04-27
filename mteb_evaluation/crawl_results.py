import mteb
from mteb.cache import ResultCache
import pandas as pd
import os
from glob import glob
import json


lang_map = {"eng": "en",
            "fra": "fr",
            "fin": "fi",
            "zho": "zh",
            "ita": "it",
            "deu": "de",
            "vie": "vi",
            "jpn": "ja",
            "ukr": "uk",
            "ell": "el",
            "hin": "hi",
            "bul": "bg",
            "fas": "fa"}

wanted_languages_tatoeba = [f"{lang}-eng" for lang in lang_map.keys()]
wanted_languages_rte = ["en", "fr", "de", "it"]
wanted_languages_webf = [f"{lang}" for lang in lang_map.keys()]
wanted_languages = wanted_languages_rte + wanted_languages_tatoeba + wanted_languages_webf

selected_tasks = [#("Retrieval", "ARCChallenge"),
                  #("Summarization", "SummEvalSummarization.v2"),
                  #("Summarization", "SummEvalFr"),
                  #("BitextMining", "Tatoeba")
                  ("PairClassification", "RTE3")
                  #("Retrieval", "WebFAQRetrieval")
                  ]

model_names = ["openai__text-embedding-3-small", "openai__text-embedding-3-large", "google__gemini-embedding-001"] + glob("/flash/project_462000883/models/*")
model_names = [os.path.basename(m).replace("__", "/") for m in model_names]
model_names += [m.split("/")[-1] for m in model_names]   # checking for different formats...?

hf_cache=os.environ["HF_HOME"]
#print(hf_cache)


cache_path=f"{hf_cache}/mteb"
cache = ResultCache(cache_path=cache_path)
if not os.path.exists(cache_path):
    # run this the first time
    cache.download_from_remote()

def read_results(task_type, task_name, model_names):
    tasks = mteb.get_tasks(task_types=[task_type])
    

    results = cache.load_results(
        models=model_names,
        tasks=tasks,
        include_remote=True, # default
    )
    #assert , f"Could not read any results for {task_type}:{task_name}"

    parsed_results = {}
    for r in results:
        model = r.model_name
        for t in r.task_results:
            if t.task_name == task_name:
                if len(t.scores["test"]) == 1:
                    #print(t.scores["test"][0]["main_score"])
                    if model not in parsed_results.keys():
                        parsed_results[model] = t.scores["test"][0]["main_score"]
                    else:
                        print("duplicate results")
                else:
                    for sub_t in t.scores["test"]:
                        if f"{model}_{sub_t['hf_subset']}" not in parsed_results.keys():
                            if sub_t['hf_subset'] in wanted_languages:
                                #print(f"here with {sub_t['hf_subset']}")
                                parsed_results[f"{model}_{sub_t['hf_subset']}"] = sub_t["main_score"]
                        else:
                            print("duplicate results")
                    sub_t = "missing"  # no leakage to next

    return parsed_results

for task_type, task_name in selected_tasks:
    
    parsed_results = read_results(task_type, task_name, model_names)
    #print(parsed_results)
    #print(len(parsed_results))
    print(json.dumps(parsed_results))
    #if parsed_results == {}:
    #    print(f"Task {task_name} All results missing")
    #else:
    #    print(f"{task_name} missing results for:")
    #    print(set(model_names) - set(parsed_results.keys()))
    #print(model_names)
