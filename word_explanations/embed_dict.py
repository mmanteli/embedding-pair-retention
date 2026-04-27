from data_and_encoder import load_model, embed, write_dataset
import sentence_transformers
import torch
import datasets
import os
from jsonargparse import ArgumentParser
import json
import copy
import numpy as np
import pickle
from reread_and_annotate_plots import run_annotation


# set this to true with parameter --debug
debug = False
def report(msg):
    if debug:
        print(msg)


def read_data(path):
    if ".pkl" in path:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return datasets.load_from_disk(path)

def write_json(j, save_path):
    with open(save_path, "w") as f:
        json.dump(j,f)

def embed_dictionaries(options):
    model = None
    for d_path in options.dictionary:
        lang, _ = os.path.splitext(os.path.basename(d_path))
        embedding_save_path = options.embedding_save_path(lang)
        if os.path.exists(embedding_save_path):
            print("Reading precalculated embeddings")
            ds = read_data(embedding_save_path)
            yield lang, ds
        else:
            if model is None:
                model = load_model(options.model_name)
                print("Loaded the embedding model")
            ds = datasets.load_dataset("text", data_files={"train": d_path})
            embedded_dictionary = embed(model, ds["train"]["text"], options).tolist()
            ds["train"] = ds["train"].add_column("embeddings", embedded_dictionary)
            os.makedirs(os.path.dirname(options.embedding_save_path(lang)), exist_ok=True)
            write_dataset(ds, options.embedding_save_path(lang))
            yield lang, ds


def show_top_bottom_k(texts, embeddings, labels, dim, k=10):
    """
    Sort texts and embeddings by a specific embedding dimension and show top/bottom k texts.
    
    Args:
        texts: list of text strings (length N)
        embeddings: numpy array of shape (N, dim_embeddings)
        dim: dimension index to sort by
        k: number of top/bottom texts to show
    """
    # This function is from Claude
    # 1. Make copies
    embeddings_copy = embeddings.copy()
    texts_copy = texts.copy()
    labels_copy = labels.copy()

    # 2 & 3. Sort both by the specified dimension
    sorted_indices = np.argsort(embeddings_copy[:, dim])  # ascending order
    embeddings_sorted = embeddings_copy[sorted_indices]
    texts_sorted = [texts_copy[i] for i in sorted_indices]
    labels_sorted = [labels_copy[i] for i in sorted_indices]

    if k >0:  # if we want to show something
        # 4. Show top and bottom k texts
        print(f"Sorting by dimension {dim}\n")
        print(f"--- Bottom {k} (lowest values) ---")
        for i in range(k):
            print(f"[{embeddings_sorted[i, dim]:.4f}] {texts_sorted[i]} ({labels_sorted[i]})")
        print(f"\n--- Top {k} (highest values) ---")
        for i in range(-k, 0):
            print(f"[{embeddings_sorted[i, dim]:.4f}] {texts_sorted[i]} ({labels_sorted[i]})")
        print("")

    return texts_sorted, embeddings_sorted, labels_sorted


def read_prefitted_ICA_model_(path):
    if ".pkl" in path:
        with open(path, "rb") as f:
            mixer1 = pickle.load(f)
            try:
                mixer2 = pickle.load(f)
                return mixer1, mixer2
            except:
                print("Only one prefitted model found, using that for both cases, paired and embedding difference.")
                return mixer1, mixer1
    elif ".pt" in path:
        raise NotImplementedError("No reading for .pt model implemented yet")
    return mixer1, None


def transform_data(model, data):
    try:
        return mixer.transform(embs)
    except:
        # SCA: we only have access to the matrices
        # apply by hand
        M = mixer["V"]
        if embs.shape[1] != M.shape[1]:
            raise ValueError(
                f"Dimension mismatch: embeddings have {embs.shape[1]} features, "
                f"but M expects {M.shape[1]}."
            )
        return embs@M.T


parser = ArgumentParser(prog="Mapping dictionaries to fitted ICA space.")
parser.add_argument('--model_name', '--model_path', '--model', type=str, required =True)
parser.add_argument('--dataset_name', '--dataset', type=str, required=True)
parser.add_argument('--method', type=str, required =True)
parser.add_argument('--dim', type=int|float, required=True)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--dictionary', nargs="+", default=["english"])
parser.add_argument('--show', default=0, type=int, help="number of examples shown in printout.")
parser.add_argument('--debug', action='store_true', help="verbosity etc.")
parser.add_argument('--prompt', '--prompted', action="store_true", help="Look for results with prompt.")
options = parser.parse_args()
debug = options.debug  # set the "global" variable

# parse these so it's possible to only give the languages
for i, dct in enumerate(options.dictionary):
    if not os.path.exists(dct):
        options.dictionary[i] = f"dictionary_analysis/{dct}.txt"
        assert os.path.exists(options.dictionary[i]), f"Cannot find or construct a valid dictionary path for {dct}"

# remove trailing /
if options.model_name[-1] == "/":
    options.model_name = options.model_name[:-1]

# redirect to local path if possible
if os.path.exists(options.model_name):
    # local model
    options.parsed_model_name = os.path.basename(options.model_name)
    report(f"Parsed model name: {options.parsed_model_name}\nModel path: {options.model_name}")
else:
    if os.path.exists(f"/flash/project_462000883/models/{options.model_name}"):
        # local model again
        options.parsed_model_name = options.model_name
        options.model_name = f"/flash/project_462000883/models/{options.model_name}"
        report(f"Parsed model name: {options.parsed_model_name}\nModel path: {options.model_name}")
    else:
        # load from hf hub, parse the name to match results
        options.parsed_model_name = options.model_name.replace("/", "__",)
        report(f"Parsed model name: {options.parsed_model_name}\nModel path (from hf-hub): {options.model_name}")

# paths for stats and fitted model
if options.prompt:
    mixer_path = f"fitted_models_with_prompt/{options.method}_{options.dim}/{options.dataset_name}/{options.parsed_model_name}.pkl"
    peak_stats_path = f"stats_with_prompt/{options.method}_{options.dim}/{options.dataset_name}/{options.parsed_model_name}.json"
    dict_lang = "-".join([os.path.basename(i).replace(".txt", "") for i in options.dictionary])
    options.embedding_save_path = lambda x: f"embeddings/dictionaries/{x}/{options.parsed_model_name}.hf"
    save_results_path = lambda x: f"results_with_prompt/{options.method}_{options.dim}/dictionaries/{options.dataset_name}/{x}/{options.parsed_model_name}.hf"
else:
    mixer_path = f"fitted_models/{options.method}_{options.dim}/{options.dataset_name}/{options.parsed_model_name}.pkl"
    peak_stats_path = f"stats/{options.method}_{options.dim}/{options.dataset_name}/{options.parsed_model_name}.json"
    dict_lang = "-".join([os.path.basename(i).replace(".txt", "") for i in options.dictionary])
    options.embedding_save_path = lambda x: f"embeddings/dictionaries/{x}/{options.parsed_model_name}.hf"
    save_results_path = lambda x: f"results/{options.method}_{options.dim}/dictionaries/{options.dataset_name}/{x}/{options.parsed_model_name}.hf"


# read the stats file, containing the peak indices
#with open(peak_stats_path,'r') as f:
#    j = json.load(f)

#peak_indices = j["peak_indices"]
#print(f"Peaks for this model/dataset are {peak_indices}")


if not torch.cuda.is_available():
    print("Run this on GPU only, for now.")
    exit()

print("Loading mixer")
mixer, _ = read_prefitted_ICA_model_(mixer_path)

print("Embedding + transforming")
transformed = []
texts = []
labels = []
langs = []
for lang, ds in embed_dictionaries(options):
    langs.append(lang)
    embs = np.array(ds["train"]["embeddings"]) 
    texts.extend(ds["train"]["text"])
    labels.extend([lang for i in range(len(ds["train"]["text"]))])
    report(f"Embedded shape: {embs.shape}")
    report(f"Length of labels {len(labels)}")
    transformed.append(transform_data(mixer, embs))
    report(f"Transformed shape: {transformed[-1].shape}")

lang_combined = "-".join(langs)
transformed = np.vstack(transformed)
assert len(transformed) == len(texts), f"len(transformed) == len(texts), {len(transformed)} == {len(texts)}"
assert len(texts) == len(labels), f"len(labels) == len(texts), {len(labels)} == {len(texts)}"
report(f"Transformed shape after hstack {transformed.shape}")
results = {}
for peak in range(options.dim):
    texts_sorted, embeddings_sorted, labels_sorted = show_top_bottom_k(texts, transformed, labels, peak, k = options.show)
    results[str(peak)] = {"texts": texts_sorted, "labels": labels_sorted, "embs_sorted": embeddings_sorted.tolist()}


dd = datasets.DatasetDict({split: datasets.Dataset.from_dict(data) for split, data in results.items()})
report(f"Final dataset")
report(dd)
pth = save_results_path(lang_combined)
os.makedirs(os.path.dirname(pth), exist_ok=True)
write_dataset(dd, pth)

print("ANNOTATING")
# annotate figures
original_figure_path = f"stats{'_with_prompt' if options.prompt else ''}/{options.method}_{options.dim}/{options.dataset_name}/{options.parsed_model_name}_fig.html"
assert os.path.exists(original_figure_path)
run_annotation(original_figure_path, pth)

