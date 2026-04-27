# embedding-pair-retention
Using kNN overlap and ICA to analyze structural retention in embedding evaluation tasks


## Running

### The running the base experiment

- Run ``fit_paired_data.py`` with appropriate parameters (see ``arguments.py`` for them). This creates:
    - embeddings, if you specify a saving location
    - fitted ICA model
    - plot and stats associated with the run
- Run ``run_knn.py`` to use the embeddings produced in the previous step to calculate neighborhood retention.

### Running stability

- To run stability, follow example set ``by run_stability.sh``. Running it from the current directory & parameters won't work. ``--random_init`` in ``arguments.py`` is the key.

### Word explanations

- Download vocabularies from the address specified in ``word_explanations/README.md``
- Follow the examples in ``embed_dict.py`` and ``embed_with_closed_weight.py``


## Configs

Any time ``arguments.py`` is used, you can give the columns, labels, number of examples, prompt, and so on, in a clean configuration file. Examples in ``configs/``.


## Citation and acknowledgements

[Deleted for review]