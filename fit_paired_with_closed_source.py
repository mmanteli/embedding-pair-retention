from sklearn.decomposition import FastICA, PCA
from sca.models import SCA, SCANonlinear
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Value
import datasets  # bad, i know i know
import pandas as pd
import numpy as np
import torch
import random
import pickle
import os
from os import makedirs, listdir
from os.path import exists, dirname, isfile, basename, splitext
import copy
from pprint import pp as prettyprint
from data_and_encoder import read_paired_dataset, write_dataset, read_precalculated_embeddings
from fit_paired_data import save_fitted_models
from evaluate import calculate_statistics, save_statistics, plot_the_bar_graph, plot_correlation_heatmap
from arguments import parser, parse_further

DO_NOT_OVERWRITE=True
global seed
global debug  # controls verbosity
debug=False
# debug messages
def report(message):
    if debug:
        print(message, flush=True)


from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

from google import genai
client_gemini = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

def get_embedding_openai(texts, model="text-embedding-3-small", chunk_size=500):
    embeddings = []
    if isinstance(texts, str):
        texts = [texts]
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        result = client.embeddings.create(input=chunk, model=model)
        embeddings += [e.embedding for e in result.data]
    return embeddings



def get_embedding_gemini(texts, model="gemini-embedding-001", chunk_size=100):
    # force bc I'm getting tired.... TODO as they say
    chunk_size=90
    embeddings = []
    if isinstance(texts, str):
        texts = [texts]
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        result = client_gemini.models.embed_content(model=model, 
                                             contents=chunk,
                                             #config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")   # this would be interesting
                                             )
        embeddings += [e.values for e in result.embeddings]
        
    return embeddings

def get_embeddings(text, model):
    print(f"Trying to embed. Example of text is: {text[0][:100] if isinstance(text, list) else text[:100]}")  # sanity
    if model == "gemini-embedding-001":
        return get_embedding_gemini(text, model)
    else:
        return get_embedding_openai(text, model)

def init_model(method, num_components, random_init):
    """Initialize a ICA, SCA, nonlinear-SCA or PCA"""

    if random_init:
        # this makes it possible to have deterministic data split but non-deterministic model init
        random_state = np.random.default_rng().integers(0,100000)
        # you can also set these for SCA, however rand is recommended here
        random_state_SCA = "rand"
    else:
        random_state = seed
        random_state_SCA = "pca"
    if method == "ICA":
        return FastICA(n_components=num_components,
                       random_state=random_state,
                       whiten='unit-variance', 
                       max_iter=500000)
    elif method == "SCA":
        return SCA(n_components=num_components,
                   n_epochs=10000,
                   init=random_state_SCA,   # rand recommended for > 1 population, which we assume to have
                   lam_sparse=0.005)  # recommended by devs to be between 1e-2 and 1e-1
    elif method == "nonlinear-SCA":
        return SCANonlinear(n_components=num_components,
                            n_epochs=10000,
                            init=random_state_SCA,   # rand recommended for > 1 population, which we assume to have
                            lam_sparse=0.005) # recommended by devs to be between 1e-2 and 1e-1
    elif method == "PCA":
        # fully deterministic
        return PCA(n_components=num_components)
    else:
        raise NotImplementedError()

def fit_model(method, X, num_components, random_init=False):
    """
    Apply I/SCA transformation to a collection of observations X, assuming n_components sources.
    If X given as a tuple, it will be concatented for fitting and then transformed separately.
    Returns estimated matrix and the transformer=object containing mixing and unmixing matrices.
    """

    # initialize new model
    report("Initializing a new model...")
    transformer = init_model(method, num_components, random_init)

    # create shuffled fitting data:
    if isinstance(X, tuple):   # paired dataset: fit together, transform separately
        report("Fitting the model with a paired dataset.")
        Z = np.concatenate(X)
        np.random.shuffle(Z)
        assert set(map(tuple, Z)) == set(map(tuple,np.concatenate(X))), "Shuffled the wrong dimension; embeddings might be transposed."
    else:
        report("Fitting the model on the embedding difference.")
        Z = copy.deepcopy(X)
        np.random.shuffle(Z)
    
    assert num_components <= Z.shape[0], f"Too few data instances given to do fitting with {num_components}"
    # fit the model with Z
    transformer.fit(Z)

    # transform the data:
    if isinstance(X, tuple):
        X_estimated = transformer.transform(X[0])
        Y_estimated = transformer.transform(X[1])
        report("Model fitted and paired data transformed.")
        return (X_estimated, Y_estimated), transformer
    else:
        X_estimated = transformer.transform(X)
        report("Model fitted and single data transformed.")
        return X_estimated, transformer


def main(options):

    if not exists(options.embs):
        print("Embedding with OpenAI models")
        ds =  read_paired_dataset(options)
        
        # add prompt:
        if options.prompt: # We want to add a text based prompt (i.e. not in model definition to have more control)
            report(f"Adding text promt: {options.prompt} with template Instruct: prompt\nQuery: text_passage")
            ds = ds.map(lambda x: {options.columns[0]: f"Instruct: {options.prompt}\nQuery: {x[options.columns[0]]}"} )
        
        # get these separately to do batching
        texts_0 = ds[options.columns[0]]
        texts_1 = ds[options.columns[1]]
        # embed
        d1 = get_embeddings(texts_0, model=options.model_name)
        d2 = get_embeddings(texts_1, model=options.model_name)
        # add to ds
        ds = ds.add_column(options.embedding_column_names[0], d1)
        ds = ds.add_column(options.embedding_column_names[1], d2)
        # save
        write_dataset(ds, options.embs)
        d1 = np.array(d1) #ds[options.embedding_column_names[0]])
        d2 = np.array(d2) #ds[options.embedding_column_names[1]])
        # check that we did not flatten in the wrong direction
        assert all([all(d1[i] == np.array(ds[options.embedding_column_names[0]][i])) for i in range(len(ds))]), "most likely flattened in wrong dim in ds.add_column()"
        assert all([all(d2[i] == np.array(ds[options.embedding_column_names[1]][i])) for i in range(len(ds))]), "most likely flattened in wrong dim in ds.add_column()"
    else:
        print("Reading pre-calculated embeddings")
        d1, d2, ds, _ = read_precalculated_embeddings(options.embs, options.embedding_column_names)

    print("Fitting a new model with paired embeddings and another one with their differences.")
    # fit model with required dimension
    (A,B), mixer1 = fit_model(options.method, (d1,d2), options.dim, random_init=options.random_init)
    report(f"Resulting shape of fitted-transformed embeddings is {A.shape}")
    # fit a similar model with the embedding differences
    sentence_differences = d1-d2
    A_B, mixer2 = fit_model(options.method, sentence_differences, options.dim, random_init=options.random_init)
    if options.save_fitted:
        save_fitted_models(mixer1, mixer2, options)

    ISCA_differences = A-B   # method(emb1)-method(emb2), as opposed to A_B = method(embd1-emb2)
    C = B.copy()
    np.random.shuffle(C)
    ISCA_differences_shuffled = A - C  # shuffled to validate findings

    prefix = options.method
    # collect results
    d = {#"emb1": d1,  # these already exist, along with original texts
         #"emb2": d2,
         "emb1-emb2": sentence_differences,
         f"{prefix}1": A,
         f"{prefix}2": B,
         f"{prefix}1-{prefix}2": ISCA_differences,
         f"{prefix}1-shuffled({prefix}2)": ISCA_differences_shuffled,
         f"{prefix}(emb1-emb2)": A_B
        }

    # add columns to the dataset
    for k,v in d.items():
        ds = ds.add_column(k, v.tolist())
    
    # save results
    makedirs(dirname(options.result_path), exist_ok=True)
    write_dataset(ds, options.result_path)

    if options.stat_path:
        if options.data_split:
            options.data_name += f":{options.data_split}"
        print("Calculating and saving statistics...")
        makedirs(dirname(options.stat_path), exist_ok=True)
        peaks, j = calculate_statistics(ds, options.method)
        j["peak_indices"] = peaks.tolist()
        save_statistics(j, options.stat_path)
        fig = plot_the_bar_graph(ds, options.method, basename(options.data_name), basename(options.model_name))
        base, ext = splitext(options.stat_path)
        fig.write_html(base+"_fig.html")
        m = mixer1.components_ if options.method != "SCA" else mixer1.params["V"]
        fig2 = plot_correlation_heatmap(m, peaks, title=f"{options.method}: {basename(options.data_name)}, {basename(options.model_name)}")
        fig2.savefig(base+"_matrix.png")

    print("Done.")
    







if __name__=="__main__":
    options = parser.parse_args()
    global seed
    seed = options.seed
    random.seed(seed)
    np.random.seed(seed)
    debug = options.debug
    options = parse_further(options) # check that values are valid, create missing values from defaults
    assert options.model_name in ["test-openai", "text-embedding-3-large", "text-embedding-3-small", "gemini-embedding-001"], f"Invalid model for OpenAI embedding {options.model_name}"

    if options.prompt:
        assert "_with_prompt" in options.embs
        assert "_with_prompt" in options.save_fitted
        assert "_with_prompt" in options.result_path
        assert "_with_prompt" in options.stat_path
    # check if we already have results...
    if (isfile(options.result_path) or (exists(options.result_path) and ".hf" in options.result_path and len(listdir(options.result_path))>0)) and DO_NOT_OVERWRITE:
        print("Exiting, since result file exists already. Change global DO_NOT_OVERWRITE do disable this.")
        exit()
    print(f"Starting OpenAI/Gemini embedding", flush=True)
    prettyprint(vars(options))
    main(options)
    