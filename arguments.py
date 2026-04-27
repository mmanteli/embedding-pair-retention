from jsonargparse import ArgumentParser, ActionConfigFile
from ast import literal_eval
from pprint import pp as prettyprint
from os.path import exists, splitext



# Arguments defined here

parser = ArgumentParser(prog="Fit I/S/PCA for paired data instances")
parser.add_argument('--config', action=ActionConfigFile)
method_options = parser.add_argument_group("Method parameters")
data_options = parser.add_argument_group("Data parameters")
embedder_options = parser.add_argument_group("Embedder/Encoder parameters")
saving_options = parser.add_argument_group("Saving parameters")

# method: ICA, SCA, PCA, etc
method_options.add_argument('--method', choices=["ICA", "SCA", "nonlinear-SCA", "PCA"],
                    help="Which method to use.")
method_options.add_argument('--dim', '--num_components', type=int|float,
                    help='How many components assumed in I/SCA. 0 implies model\'s embedding dimension, (0,1) implies level of variance optimisation.')
method_options.add_argument('--prefitted_model', type=str, default=None,
                    help="Location for pretrained ICA model that will be used. NOTE that fully saving and SCA model is currently impossible.\
                        \nUsng this sets other model parameters to None.")
method_options.add_argument('--random_init', action="store_true",
                    help='Dataset filtering and shuffling uses seed; whether to use the same seed for the I/SCA model or not. Usefull for stability testing: same data, different model.')

# data
data_options.add_argument('--data_name', '--data', type=str,
                    help="Huggingface dataset name/path to the dataset. Colon after name implies split (see --data_split).")
data_options.add_argument('--train_validation_test_split', type=str, default=None,
                    help="If given and if data is a DatasetDict, this split is selected, otherwise selected in order train>validation>test.")
data_options.add_argument('--data_split', '--split',type=str,
                    help="Split (i.e. language, other subselection, NOT train-test etc.) of the data to be used, can also be specified with colon separation from the --data_name parameter.")
data_options.add_argument('--columns','--data_keys', type=str,
                    help="keys/column names to read fields from given dataset. Give as string-tuple or colon separated, see parse_further().")
data_options.add_argument('--labels', '--select_labels', type=str,
                    help="Column name and which values to NOT drop from the dataset. Give as string-tuple or colon separated, see parse_further().")
data_options.add_argument('--downsample', type=int, default=None,
                    help="downsample large data to a give number of instances.")

# embedding model
embedder_options.add_argument('--model_name','--model', type=str, 
                    help="Model name on Huggingface or path to the model to be used to encode/embed data.")
embedder_options.add_argument('--batch_size', type=int,
                    help="Model inner batch size", default=8)
# no query used here by default: Otherwise, I/SCA will latch onto that -> results we do not want
embedder_options.add_argument('--task', '--prompt_name', default=None, choices=["STS","Summarization","BitextMining","Retrieval"], 
                    help='Which query/promt to use from model presets. None == no query. Almost always, do NOT use prompt, as I/SCA will fit that as well.')
embedder_options.add_argument('--prompt', default=None, type=str, 
                    help='Text promt template to be prepended to the input texts in format Instruct: <prompt>\nQuery: <text instance>')
embedder_options.add_argument('--embs','--embeddings', type=str, default=None,
                    help="Saving location for calculated embeddings, end in .pkl for pickling and .hf for huggingface. Additionally, if this exists, it is read and no recalculation is done.")
embedder_options.add_argument('--embedding_column_names', type=str|tuple, default=("emb1", "emb2"),
                    help="Which columns to read from precalculated embeddings, give as str(tuple). Only for reading embeddings, written with default values.")


# saving
saving_options.add_argument('--save_fitted', type=str, default=None,
                    help="Location for saving the paired-data ICA model or the matrices of SCA model.")          
saving_options.add_argument('--result_path','--results', type=str, default=None,
                    help="path for saving ICA results, a defaul path starting with results/ created if not given.")
saving_options.add_argument('--stat_path','--stats', type=str|bool, default=None,
                    help="Path for saving statistics, or simply True to create a default path (same as results but with stats/)")

parser.add_argument('--seed', default=42, type=int,
                    help="Seed for all processes, except with --random_init. SCA requires different handling, see run_stability.py")
parser.add_argument('--debug', action="store_true",
                    help="Verbosity etc.")


def eval_tuple(c):
    """
    Evaluate a parameter that has multiple values (like dataset:task or (columns,value_to_choose))
    Two formats as spaces in some column names can cause problems.
    """
    if ":" in c:
        # parse a colon notation, remove quotes if needed
        c_ = c.replace("\"", "").split(":")
        assert len(c_) == 2, f"Trying to evaluate tuple with colon notation, too many values found: {c}."
        return (c_[0], c_[1])
    # parse a "normal" tuple
    assert c[0] == "(" and c[-1] == ")" and len(c.split(",")) == 2, f"Argument {c} given incorrectly, give as a tuple in quotation marks"
    return literal_eval(c)

def parse_further(options):
    """Check that given parameters obey some rules"""

    # check that we either have method params or a pretrained model
    assert options.method and options.dim or options.prefitted_model, "Neither a prefitted model or complete model params given. Give --method and --dim or --prefitted_model"

    # check that we have data and values to read and encoder OR ready made encodings
    assert (options.data_name and options.columns and options.model_name) or (options.embs and exist(options.embs)), \
        f"Either give --data_name, --columns and --model_name or path to precalculated embeddings --embs."
    # check that "paired values" are given correctly
    options.columns = tuple(eval_tuple(options.columns))
    if options.labels:
        options.labels = tuple(eval_tuple(options.labels))
    # parse data:task
    if ":" in options.data_name:
        options.data_name, options.data_split = options.data_name.split(":")
    if isinstance(options.embedding_column_names, str):
        options.embedding_column_names = tuple(eval_tuple(options.embedding_column_names))

    # check that it is possible to use the dimension value
    assert options.dim >= 0, "I/SCA dimension --dim needs to be positive or 0. See meanings for each value with --help."
    options.dim = int(options.dim) if (options.dim >= 1 or options.dim == 0) else float(options.dim)
    
    # if prefitted model, warn if other params are given.
    if options.prefitted_model:
        # set method and dim to None; so that they cannot be accidentally used.
        options.method = None
        options.dim = None
        assert options.result_path, "Using a prefitted model requires explicit --result_path definition, please provide one."
        assert exists(options.prefitted_model), "--prefitted_model given, but it does not exist/cannot be found."
        if options.save_fitted is not None:
            print("WARNING: Cannot save a fitted model when using a prefitted model, setting --save_fit parameter to None.")
            options.save_fitted == None

    # make default result path if not given AND we're not using prefitted model => at least --method and --dim should exist.
    if options.result_path is None:
        # create a default result path 
        assert options.dim, f"Cannot construct default result saving path, --dim missing, correct or give explicit --result_path."
        dimension = str(options.dim)
        assert options.model_name, f"Cannot construct default result saving path, --model_name missing, correct or give explicit --result_path."
        if options.model_name[-1] == "/":
            options.model_name = options.model_name[:-1]  # i.e. if a local path, remove last "/" 
        parsed_model_name = basename(options.model_name)
        assert options.data_name, f"Cannot construct default result saving path, --data_name missing, correct or give explicit --result_path."
        parsed_data_name = ''.join([c if c.isalnum() else "-" for c in basename(options.data_name)])  # for save saving
        if options.data_split:
            parsed_data_name += ":" + options.data_split   # add split with colon notation, makes reading 
        options.result_path = f"results/{options.method}_dim_{dimension}/{parsed_data_name}__{parsed_model_name}_results.hf"
    if options.stat_path:
        if options.stat_path is True or options.stat_path == "True":
            # make default path
            if "results" in options.result_path:
                options.stat_path = options.result_path.replace("results","stats")
                base, ext = splitext(options.stat_path)
                if ext in [".tsv", ".csv", ".hf", ".jsonl"]:
                    options.stat_path=options.stat_path.replace(ext, ".json")
            else:
                if options.prompt:
                    options.stat_path = f"stats_with_prompt/{options.method}_dim_{dimension}/{parsed_data_name}__{parsed_model_name}_stats.json"
                else:
                    options.stat_path = f"stats/{options.method}_dim_{dimension}/{parsed_data_name}__{parsed_model_name}_stats.json"

    return options


# main for testing
if __name__ == "__main__":
    options = parser.parse_args()
    options = parse_further(options)
    prettyprint(vars(options))