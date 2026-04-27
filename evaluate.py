import pickle
import datasets
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import kurtosis, skew
import sys
import json
from os.path import exists, dirname, isfile, splitext
from os import makedirs, stat, remove
from datasets import load_dataset, load_from_disk
import seaborn as sns
from jsonargparse import ArgumentParser
import matplotlib.pyplot as plt


def read_results(path):
     # try reading with multiple formats
    if ".jsonl" in path:
        ds = load_dataset("json", data_files=path)
    elif ".tsv" in path:
        ds = load_dataset("csv", delimiter = "\t", data_files=path)
    elif ".csv" in path:
        ds = load_dataset("csv", delimiter = ",", data_files=path)
    elif ".pkl" in path:
        with open(path, "rb") as f:
            ds = pickle.load(f)
    else:
        ds = load_from_disk(path)
    return ds

def load_data_and_matrix(path, matrix_path=None):
    if ".pkl" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ".hf" in path:
        data = datasets.load_from_disk(path)
    else:
        print("Cannot read file {path}, only implementations for .hf & .pkl")
        exit()
    if matrix_path is None:
        matrix_path = path.replace("results", "fitted_models")
    with open(matrix_path, "rb") as f:
        model = pickle.load(f)
    try:  # ICA
        return data, model.components_
    except:  # SCA
        return data, model["V"]


def plot_correlation_heatmap(matrix, peaky_dimensions, title="Correlation heatmap"):
    matrix = normalize(matrix, axis=1)   # each ROW
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.abs(matrix), ax=ax, yticklabels=True)
    ax.set_title(str(title))
    for yl in peaky_dimensions:  # this order for the axes
        if yl is not None:
            ylabels = ax.get_yticklabels()
            highlight_y = yl
            ylabels[highlight_y].set_fontweight('bold')
            ylabels[highlight_y].set_color('red')
    ax.figure.canvas.draw()
    return fig

def plot_the_bar_graph(d, method, data_name, model_name, show=False):

    icas = np.array(d[f"{method}1-{method}2"])
    icas_shuf = np.array(d[f"{method}1-shuffled({method}2)"])
    
    # what to plot
    means_ica = np.mean(np.abs(icas), axis = 0)
    means_ica_shuf = np.mean(np.abs(icas_shuf), axis = 0)
    std_ica = np.std(icas, axis = 0)
    std_ica_shuf = np.std(icas_shuf, axis = 0)
    assert len(means_ica) > 1, "Took the mean along the wrong axis?"

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"{data_name}, {model_name} — paired data",
            f"{data_name}, {model_name} — shuffled data",
        ]
    )

    # First plot
    fig1 = px.bar(means_ica, error_y=std_ica)
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)

    # Second plot
    fig2 = px.bar(means_ica_shuf, error_y=std_ica_shuf)
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)

    # Update layout (titles, axis labels, etc.)
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"{data_name}, {model_name}",
    )

    fig.update_yaxes(
        title_text="E[|Δ|] on given dimension, paired data points",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text=f"{method} dimension",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="E[|Δ|] on given dimension, paired data points",
        row=2, col=1
    )
    fig.update_xaxes(
        title_text=f"{method} dimension",
        row=2, col=1
    )

    if show:
        fig.show()
    return fig

def gini_coefficient_and_lorenz_curve(distribution):
    sorted_dist = np.sort(distribution)
    n = len(sorted_dist)
    cum_sum = np.cumsum(sorted_dist)
    y = cum_sum / np.sum(sorted_dist)
    x = np.arange(len(distribution)) / len(distribution)
    gini = (n-1)/n - (2 * np.sum((n - np.arange(1, n+1)) * sorted_dist)) / (n * np.sum(sorted_dist))
    # Note here: for first results, gini was calculated without (n-1)/n:
    # if this term is added in some old files, it is because of this.
    return gini, (x.tolist(),y.tolist())

def kurtosis_and_skew(distribution):
    return kurtosis(distribution, fisher=False), skew(distribution)

def coefficient_of_variation(distribution):
    return np.std(distribution) / np.mean(distribution)

def peak_to_mean_ratio(distribution):
    return np.max(distribution) / np.mean(distribution)

def peak_detection_iqr(distribution, th=3):
    assert len(distribution) > 1, f"Length of dist == {len(distribution)}, should be dim_ISPCA, likely taking the mean with wrong axis"
    q1 = np.percentile(distribution, 25)
    q3 = np.percentile(distribution, 75)
    iqr = q3 - q1
    outlier_indices = np.where(distribution > q3 + th * iqr)[0] # [0] for result is a 1-tuple
    return outlier_indices

def calculate_statistics(d, method, prints=False):
    # ICA(data1)-ICA(data2)
    icas = np.array(d[f"{method}1-{method}2"])
    # ICA(data1)- shuffled(ICA(data2))  => to differentiate semantics vs. wanted quality
    icas_shuf = np.array(d[f"{method}1-shuffled({method}2)"])
    # mean(abs(x)) from both
    icas_mean = np.mean(np.abs(icas), axis = 0)
    icas_shuf_mean = np.mean(np.abs(icas_shuf), axis = 0)
    assert len(icas_mean) > 1, "Took the mean along the wrong axis?"

    # find the peaky dimension(s)
    # depending on the application, you can also use the shuffled mean for this
    # but not 
    contributing_dimensions = peak_detection_iqr(icas_mean)
    contributing_dimensions_in_shuffled = peak_detection_iqr(icas_shuf_mean)
    if prints:
        print(f"Peaky dimensions are {contributing_dimensions}")
    # calculate metrics
    gini, lorenz_curve = gini_coefficient_and_lorenz_curve(icas_mean)
    var = coefficient_of_variation(icas_mean)
    peak = peak_to_mean_ratio(icas_mean)
    shuf_gini, shuf_lorenz_curve = gini_coefficient_and_lorenz_curve(icas_shuf_mean)
    shuf_var = coefficient_of_variation(icas_shuf_mean)
    shuf_peak = peak_to_mean_ratio(icas_shuf_mean)
    if prints:
        print(f"Metrics are:\n\tGini:{gini}\n\tVar:{var}\n\tPeak:{peak}")
    return contributing_dimensions, {"gini": gini, "var": var, "peak": peak, "shuf_gini": shuf_gini, "shuf_var": shuf_var, "shuf_peak": shuf_peak, "shuf_peak_indices": contributing_dimensions_in_shuffled.tolist()}

def save_statistics(j, save_path):
    if ".json" in save_path:
        with open(save_path, "w") as f:
            try:
                json.dump(j,f)
            except Exception as e:
                # remove the most likely partially written file
                if exists(save_path):
                    remove(save_path)
                raise(e)
    elif ".pkl" in save_path:
        with open(save_path, "wb") as f:
            pickle.dump(j,f)
    else:
        print(j)



parser = ArgumentParser(prog="Calculate stats from paired, fitted data.")
parser.add_argument("--data", help="Data to analyse", required=True, type=str)
parser.add_argument("--matrix", help="Path to matrix, if not given assumed to be same as data but with fitted_models/", type=str, default=None)
parser.add_argument("--method", required=True, type=str, choices=["ICA", "SCA", "PCA"])
parser.add_argument("--save", type=str, default=None)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--data_name", type=str, default="")


if __name__ == "__main__":
    options = parser.parse_args()
    path = options.data
    method = options.method
    assert exists(path), "Given path does not exists."
    print(f"In file {path}")
    save_path = options.save
    data_name = options.data_name
    model_name = options.model_name
    if save_path is None:
        if "results" in path:
            save_path = path.replace("results", "stats")
            base, ext = splitext(save_path)
            #if ext in [".tsv", ".csv", ".hf", ".jsonl"]:
            save_path=save_path.replace(ext, ".json")
            if isfile(save_path) and stat(save_path).st_size != 0:
                # we already have a results for this file
                print("Stopping as results for this file already exist.")
                exit()
            
            
            print(f"\tSaving to {save_path}")
        else:
            print("\tCould not construct a save path from input path, exiting...")
            exit()
    makedirs(dirname(save_path), exist_ok=True)
    d, m = load_data_and_matrix(path, matrix_path=options.matrix)
    peaks, j=calculate_statistics(d, method)
    j["peak_indices"] = peaks.tolist()
    save_statistics(j, save_path)
    fig = plot_the_bar_graph(d, method, data_name, model_name)
    base, ext = splitext(save_path)
    fig.write_html(base+"_fig.html")
    fig2 = plot_correlation_heatmap(m, peaks)
    fig2.savefig(base+"_matrix.png")
    print("\tDone for this file.")
