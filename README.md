# embedding-pair-retention
Using kNN overlap and ICA to analyze structural retention in embedding evaluation tasks


## Running

### Environment

These experiments are designed to be run in a HPC environment. As faithfull as possible requirements are given in ``requirements.txt``. 

### The running the base experiments

- Run ``fit_paired_data.py`` with appropriate parameters (see ``arguments.py`` for them). This creates:
    - embeddings, if you specify a saving location
    - fitted ICA model (again, requires a saving location)
    - plot and stats associated with the run (again, a saving location required)
- Run ``run_knn.py`` to use the embeddings produced in the previous step to calculate neighborhood retention.

To run an experiment with ICA 32 on embeddings of Tatoeba:eng-fra produced by intfloat/multilingual-e5-small, run

```bash
python fit_paired_data.py \
        --method="ICA" \
        --dim=32 \
        --model="intfloat/multilingual-e5-small" # or path to a local model
        --config="configs/no_prompt/tatoeba.json" \  # contains dataset name and which columns to read
        --split="en-fr" \   # huggingface split name
        --downsample=5000 \
        --embs=<saving location for embeddings, to avoid recalculation> \
        --save_fitted=<saving location for fitted ICA model> \
        --result_path=<path for saving the results, i.e. ICA transformed embeddings> \
        --stat_path=<path for saving Gini coefficient and plots, suffixes like "_fig.png" are added automatically> \
        --debug # more reporting
```

You can also test the "test config" in ``configs/test/test_ICA.json``.

### Running stability

- To run stability, follow example set ``by run_stability.sh``. Running it from its current directory & parameters won't work, copy it to the main dir level. ``--random_init`` in ``arguments.py`` is the key to differently initialize ICA models.

### Word explanations

- Download vocabularies from the address specified in ``word_explanations/README.md``
- Follow the examples in ``embed_dict.py`` and ``embed_with_closed_weight.py`` similar to the stability.


## Configs

Any time ``arguments.py`` is used, you can give the columns, labels, number of examples, prompt, and so on, in a clean configuration file. Examples in ``configs/``.


## Citation and acknowledgements

[Deleted for review]