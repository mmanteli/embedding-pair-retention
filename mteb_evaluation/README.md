# MTEB-evaluation

Here we present how we obtained the scores for our experiments:

- RTE3-multi was the only one with available scores for all the models. Hence, ``crawl_results.py`` crawls the results for this task.
- Others can be run with ``python mteb_evaluate.py``
- Closed models require API keys set in the environment and can be run with ``python mteb_evaluate_closed_weight.py``