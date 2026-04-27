import mteb
from mteb.cache import ResultCache
from sentence_transformers import SentenceTransformer
import sys
import os
from data_and_encoder import load_model
languages = ["fra", "zho", "ita", "deu", "jpn", "bul", "fin", "hin", "ell", "fas", "ukr", "vie", "eng"]
langauges_tatoeba = [f"{i}-eng" for i in languages if i!="eng"]


assert len(sys.argv) == 2, "Give model to evaluate"
model_name = sys.argv[1]
if os.path.exists(model_name):
    # take basename and convert
    model_name = os.path.basename(model_name).replace("__", "/")
#try:
#    model = mteb.get_model(model_name, trust_remote_code=True) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
#except:
model = load_model(model_name)
model.eval()


# example of queries
#model = SentenceTransformer(
#    "intfloat/multilingual-e5-small", 
# prompts={"query": "Query:", "document": "Passage:"}
#)

def get_tatoeba():
    # from ChatGPT
    # Get the Tatoeba task
    tatoeba = mteb.get_task("Tatoeba")
    # --- find which attribute in your installed mteb holds the list of subsets ---
    subsets = None
    for attr in ("available_subsets", "hf_subsets", "subsets"):
        if hasattr(tatoeba, attr):
            subsets = getattr(tatoeba, attr)
            break

    if subsets is None:
        raise RuntimeError(
            "Couldn't find subsets on the Tatoeba task. "
            "Print tatoeba.__dict__ (or dir(tatoeba)) to inspect available fields."
        )

    # Pick the subset names that correspond to eng-deu and eng-fin
    # (naming can vary slightly; this matches common MTEB conventions)
    wanted = langauges_tatoeba
    chosen = [s for s in subsets if s in wanted]

    if not chosen:
        raise RuntimeError(
            f"None of {wanted} found. Available subsets include e.g.: {subsets[:30]}"
        )

    # --- create a new task instance restricted to those subsets ---
    # In mteb 2.x, tasks commonly accept `task_langs` and/or `hf_subsets`/`eval_splits`.
    # For Tatoeba, restricting by hf_subsets is the usual route.
    tatoeba_restricted = mteb.get_task("Tatoeba", hf_subsets=chosen)
    return tatoeba_restricted

def get_webfaq():
    webfaq = mteb.get_task("WebFAQRetrieval")
    # --- find which attribute in your installed mteb holds the list of subsets ---
    subsets = None
    for attr in ("available_subsets", "hf_subsets", "subsets"):
        if hasattr(webfaq, attr):
            subsets = getattr(webfaq, attr)
            break

    if subsets is None:
        raise RuntimeError(
            "Couldn't find subsets on the WebFAQRetrieval task. "
            "Print webfaq.__dict__ (or dir(webfaq)) to inspect available fields."
        )

    # Pick the subset names that correspond to eng-deu and eng-fin
    # (naming can vary slightly; this matches common MTEB conventions)
    wanted = languages
    chosen = [s for s in subsets if s in wanted]

    if not chosen:
        raise RuntimeError(
            f"None of {wanted} found. Available subsets include e.g.: {subsets[:30]}"
        )

    # --- create a new task instance restricted to those subsets ---
    # In mteb 2.x, tasks commonly accept `task_langs` and/or `hf_subsets`/`eval_splits`.
    # For Tatoeba, restricting by hf_subsets is the usual route.
    webfaq_restricted = mteb.get_task("WebFAQRetrieval", hf_subsets=chosen)
    return webfaq_restricted


# Select tasks
default_tasks = mteb.get_tasks(tasks=[
                            "ARCChallenge", 
                            "SummEvalSummarization.v2",
                            #"SuperGlueBoolQ",
                            #"RTE3"
                            ], 
                            languages=languages
                            )

tatoeba_restricted = get_tatoeba()
webfaq_restricted = get_webfaq()
#print(default_tasks)
#print(tatoeba_restricted)
tasks = list(default_tasks) + [tatoeba_restricted] + [webfaq_restricted]

print(tasks)
# evaluate
outfile = f"mteb_out/{model_name.replace('/', '__')}"
os.makedirs(outfile, exist_ok=True)

cache = ResultCache(cache_path=outfile)
results = mteb.evaluate(model, tasks=tasks, cache=cache, encode_kwargs={"batch_size": 8})
print(results) # just in case