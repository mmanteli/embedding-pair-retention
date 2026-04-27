import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Value
import pickle
import random
from sentence_transformers import SentenceTransformer
# use these only if you know what you're doing
from prompt_utils import get_all_prompts
from os import makedirs
from os.path import exists, dirname
import torch
from ast import literal_eval
# for testing debug is always true
debug=True
def report(message):
    if debug:
        print(message, flush=True)


#-------------------------------------------Datasets---------------------------------------------#

def flatten_dataset(ds, predefined_split):
    if predefined_split:
        return ds[predefined_split]
    else:
        try:
            return ds["train"]
        except:
            try:
                return ds["validation"]
            except:
                return ds["test"]
                

def read_paired_dataset(options):
    """Read a dataset with 2 columns with some for of correspondence (translation etc.)"""

    data_name_or_path = options.data_name

    # try reading with multiple formats
    if ".jsonl" in data_name_or_path:
        ds = load_dataset("json", data_files=data_name_or_path)
    elif ".tsv" in data_name_or_path:
        ds = load_dataset("csv", delimiter = "\t", data_files=data_name_or_path)
    elif ".csv" in data_name_or_path:
        ds = load_dataset("csv", delimiter = ",", data_files=data_name_or_path)
    else:
        data_to_search = options.data_name
        if options.data_split:
            data_to_search += f":{options.data_split}"
        if exists(data_to_search):   # local dataset
            report(f"Loading {data_to_search} from disk")
            ds = load_from_disk(data_to_search)
        else:
            # try from huggingface
            try:
                if options.data_split:
                    # here split means language or other subset selection
                    ds = load_dataset(options.data_name, options.data_split)
                    report("Loaded dataset from huggingface hub")
                else:
                    ds = load_dataset(options.data_name)
                    report("Loaded dataset from huggingface hub")
            except:
                raise Exception(f"Cannot download dataset {options.data_name} from local disk nor from from Huggingface Hub.")
    report(f"Dataset read:\n{ds}")

    #try to flatten, --train_validation_test_split contains the split, if not, train>validation>test
    if isinstance(ds, DatasetDict):
        ds = flatten_dataset(ds, options.train_validation_test_split)
    

    # check whether it is in the correct format
    assert not isinstance(ds, DatasetDict), f"Dataset is still in DatasetDict format, please flatten it, for example by adding a key to --train_validation_test_split: \n{ds}"
    assert len(ds) > 0, f"Read an empty dataset"
    assert options.columns[0] in ds.column_names and options.columns[0] in ds.column_names, f"Given columns {options.columns} not found in dataset column names:\n{ds.column_names}."

    # if defined, select rows with --labels
    if options.labels:
        report(f"Filtering the dataset based on {options.labels}...")
        label_column, accepted_value = options.labels
        assert label_column in ds.column_names, f"Label column given, but no such column exists: {label_column} vs {ds.column_names}"
        if accepted_value not in set(np.unique(ds[label_column])):
            # it might be an integer/boolean mixup, try to parse that:
            accepted_value = literal_eval(accepted_value)
            assert accepted_value in set(np.unique(ds[label_column])), f"Could not find any {accepted_value} values in column {label_column}."
        ds = ds.filter(lambda x: x[label_column]==accepted_value)
        report(f"After filtering:\n{ds}")
        assert len(ds) > 0, f"Filtering with {options.labels} resulted in empty dataset."
    
    #downsample, if defined by params:
    total_rows = len(ds)
    if options.downsample and options.downsample < total_rows:
        report("Downsampling the dataset...")
        # generate a list of random indices
        random_indices = random.sample(range(total_rows), options.downsample)
        # select the rows using the random indices
        ds = ds.select(random_indices)

    # assign all values to strings (as we will be embedding them)
    # this for datasets with numerical labels; NOTE that using them might not be wise 
    # as the embeddings are likely bad for a single number
    # consider transforming them into "this sentence is negative" ect.
    for feat, typ in ds.features.items():
        if feat in options.columns and typ != Value('string'):   # i.e. anything we embed -> to string
            ds = ds.cast_column(feat, Value('string'))
    
    print(f"Dataset loaded, after row-selection, downsamplic and other filtering:\n{ds}")
    # all done
    return ds


#----------------------------------------Embedding models------------------------------------------#

def load_model(model_name, prompt_name=None):
    if prompt_name is None:
        if "llama-embed-nemotron-8b" in model_name:
            model = SentenceTransformer(
                                        model_name,
                                        trust_remote_code=True,
                                        device = "cuda:0" if torch.cuda.is_available() else "cpu",
                                        model_kwargs={"attn_implementation": "eager"}
                                        )
        else:
            model = SentenceTransformer(
                                        model_name,
                                        trust_remote_code=True,
                                        device = "cuda:0" if torch.cuda.is_available() else "cpu",  # here the default is the sdpa attention
                                        )
    else: # we generally do not want a prompt
        model = SentenceTransformer(
                                    model_name,
                                    prompts=get_all_prompts(),
                                    default_prompt_name=prompt_name,
                                    trust_remote_code=True,
                                    device = "cuda:0" if torch.cuda.is_available() else "cpu",
                                    )
    # add here any special cases
    if "gte-Qwen2-7B-instruct" in model_name:
        model.max_seq_length = 8192
    print("Model loaded")
    return model


def embed(model, input_texts, options):
    """Embed given texts with given model."""
    # Check what kind of prompting your model uses if you use prompts
    # Here we do not want prompts, and then I/SCA would only focus on that
    # But for future reference
    report(f"Embedding, example of input text is: \n{input_texts[0]}")
    return model.encode(input_texts,
                        convert_to_tensor=False,
                        normalize_embeddings=True,
                        batch_size=options.batch_size)


#---------------------------------Reading and writing embeddings-----------------------------------#

def read_pickled_data(path):
    """Read embeddings that are pickled hf-dataset"""
    report(f'Reading embeddings form {path}...')
    with open(path, "rb") as f:
        data = pickle.load(f)
    # flatten in case
    if isinstance(data, DatasetDict):
        data = flatten_dataset(data)
    if not isinstance(data, Dataset):
        raise NotImplementedError("Pickled data is assumed to be a huggingface dataset.")
    return data

def read_formatted_data(path):
    """Read embeddings from common filetypes"""
    report(f'Reading embeddings form {path}...')
    if ".jsonl" in data_name_or_path:
        ds = load_dataset("json", data_files=data_name_or_path)
    elif ".tsv" in data_name_or_path:
        ds = load_dataset("csv", delimiter = "\t", data_files=data_name_or_path)
    elif ".csv" in data_name_or_path:
        ds = load_dataset("csv", delimiter = ",", data_files=data_name_or_path)
    else: # try from huggingface or disk in the hf format
        try:
            ds = load_from_disk(path)
        except:
            try:
                ds = load_dataset(path)
            except:
                raise NotImplementedError("Cannot read embeddings from {path}, file type can be .pkl .csv .tsv .jsonl .npy .hf")
    if isinstance(data, DatasetDict):
        data = flatten_dataset(data)
    return ds



def write_pkl(data, path):
    report(f"Writing pickled data to {path}")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    report("Writing done")
    

def write_hf(ds, path):
    report(f"Saving dataset to Hugging Face format in {path}")
    ds.save_to_disk(str(path))
    report("Writing done")

def write_dataset(ds, path):
    if ".pkl" in path:
        write_pkl(ds, path)
    else:
        write_hf(ds, path)


#-----------------------------------Encoding "runners"-------------------------------------#


def read_precalculated_embeddings(path, columns):
    """Read a file containing a dataset with texts and precalculated embeddings."""
    if ".pkl" in path:
        ds = read_pickled_data(path)
    else:
        ds = read_formatted_data(path)
    t1 = np.array(ds[columns[0]])
    t2 = np.array(ds[columns[1]])
    print(f'Read pre-calculated embeddings, shape: {t1.shape}, {t2.shape}')
    return t1, t2, ds, None  # None for model, we never downloaded it


def read_and_embed_paired_dataset(options):
    """Read a dataset and embed text instances. If --embs given, write them as well."""
    ds = read_paired_dataset(options)
    model = load_model(options.model_name)
    # encode in batches:
    # the model also has a batch size, so why select batches here?
    # especially since these batches do not concern seqlen
    # -> if not some batching here a small GPU will be out of memory
    embeds1 = []
    embeds2 = []
    key1, key2 = options.columns
    
    if options.prompt: # We want to add a text based prompt (i.e. not in model definition to have more control)
        report(f"Adding text promt: {options.prompt} with template Instruct: prompt\nQuery: text_passage")
        ds = ds.map(lambda x: {key1: f"Instruct: {options.prompt}\nQuery: {x[key1]}"} )

    report("Starting embedding/encoding...")
    for i in range(0, len(ds), options.batch_size):
        batch = ds[i:i+options.batch_size]
        embeds1.extend(embed(model, batch[key1], options))
        embeds2.extend(embed(model, batch[key2], options))
    
    # add them to the dataset
    ds = ds.add_column(f"emb1", embeds1)
    ds = ds.add_column(f"emb2", embeds2)
    embeds1 = np.array(embeds1)
    embeds2 = np.array(embeds2)
    print(f'Embedding done. Shape of embedded sentences: {embeds1.shape}, {embeds2.shape}')
    if options.embs is not None:
        report(f"Writing embeddings to {options.embs}")
        makedirs(dirname(options.embs), exist_ok=True)
        write_dataset(ds, options.embs)
    return embeds1, embeds2, ds, model



if __name__=="__main__":
    pass