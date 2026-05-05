# Analysis of results

- ``analysis.ipynb`` produces the correlation tables.
- ``analysis_plotting.ipynb`` produces the correlation tables as well as plotting results.
- ``shuffling_differences.ipynb`` produces the shuffling figures and associated correlation coeffs.
- ``effect_of_k_in_kNN.ipynb`` calculates the optimal value for k (see below)


## Notes and explanations

There are two noteworthy things:

1. At first, gini-coefficient was calculated accidentally without the (dim-1)/dim term. Hence, it is added in all notebooks. Since all values missed this term, it has no effect on correlation values. Nevertheless, values in the paper include the missing term. The evaluation script has this error corrected but it is kept in the analysis-files.
2. In ``effect_of_k_in_kNN.ipynb``, the sorting was originally ascending and not descending as it should have been. Therefore, there is a correction in the analysis-files in the commit where this file was added.