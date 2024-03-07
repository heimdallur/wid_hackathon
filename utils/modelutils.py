import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import random as rnd
from datetime import datetime


def undersample_df(df, target="target", scale=0.5):
    target_bal = df[target].value_counts()
    class_min_n = target_bal.min()
    class_min = target_bal.\
        index[target_bal == class_min_n].to_list()

    dfs = [df.query(f"{target} in {class_min}")]
    class_maj = list(set(df[target].unique()) - set(class_min))
    scale_factor = int((1/scale)-1)

    for c in class_maj:
        df_us = df.\
            query(f"{target} == {c}").\
            sample(n=class_min_n*scale_factor)
        dfs.append(df_us)
    df_bal = pd.concat(dfs)

    return df_bal, class_min, class_min_n


def plot_roc(false_pos_rate, true_pos_rate, auc_value=None, model_name=""):

    plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig_roc = plt.figure()
    plt.plot(false_pos_rate, true_pos_rate)

    roc_random = np.linspace(0, 1, 100)
    plt.plot(roc_random, roc_random, color="orange", linestyle="dashed")

    plt.grid(visible=True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.figtext(0.99, 0.01, plot_timestamp, ha="right",
                fontsize=8, alpha=0.3)

    if auc_value is None:
        if len(model_name) > 0:
            plt.title(f"{model_name} - ROC Curve")
        else:
            plt.title("ROC Curve")
    else:
        if len(model_name) > 0:
            plt.title(f"{model_name} - ROC Curve (AUC: {auc_value:.2f})")
        else:
            plt.title(f"ROC Curve (AUC: {auc_value:.2f})")

    fig_roc.set_facecolor("white")

    return fig_roc


def plot_pr(recall, precision, auc_value=None, model_name=""):

    plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fig_pr = plt.figure()
    plt.plot(recall, precision)

    pr_random_x = np.linspace(0, 1, 100)
    pr_random_y = np.linspace(0, 0, 100)
    plt.plot(pr_random_x, pr_random_y, color="orange", linestyle="dashed")

    plt.grid(visible=True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.figtext(0.99, 0.01, plot_timestamp, ha="right",
                fontsize=8, alpha=0.3)

    if auc_value is None:
        if len(model_name) > 0:
            plt.title(f"{model_name} - Precision-Recall Curve")
        else:
            plt.title("Precision-Recall Curve")
    else:
        if len(model_name) > 0:
            plt.title(f"{model_name} - Precision-Recall Curve (AUC: " +
                      f"{auc_value:.2f})")
        else:
            plt.title(f"Precision-Recall Curve (AUC: {auc_value:.2f})")

    fig_pr.set_facecolor("white")

    return fig_pr


def plot_score_dist(scores: list, model_name: str = ""):

    fig_dist = plt.figure()
    plot_title = model_name + "- Score Distribution"
    plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_scores = len(scores)
    if n_scores > 10000:
        scores = rnd.sample(scores, 10000)
        subtitle = (f"10k sample of {n_scores:,} observations plotted " +
                    "for efficiency")
    else:
        subtitle = f"Based on {n_scores:,} observations"

    scores.sort()
    score_ind = list(range(len(scores)))
    score_ind_scale = [float(x)/max(score_ind) for x in score_ind]

    plt.plot(score_ind_scale, scores)

    plt.grid(visible=True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Proportion of Customers")
    plt.ylabel("Score")
    plt.suptitle(plot_title)
    plt.title(subtitle, size=10, alpha=0.3)
    plt.figtext(0.99, 0.01, plot_timestamp, ha="right",
                fontsize=8, alpha=0.3)

    fig_dist.set_facecolor("white")

    return fig_dist


def plotly_score_dist(scores: list, model_name: str = ""):

    scores.sort()
    score_ind = list(range(len(scores)))
    score_ind_scale = [float(x)/max(score_ind) for x in score_ind]

    df_plot = pd.DataFrame({"Score": scores,
                            "Proportion of customers": score_ind_scale})

    fig_score = px.line(df_plot,
                        x="Proportion of customers", y="Score",
                        title=model_name)
    fig_score.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig_score


def mean_abs_shap(shap_values, feature_names, sort_desc=True):

    mas_vals = np.abs(shap_values).mean(0)

    shap_mas_df = pd.DataFrame(
        list(zip(feature_names, mas_vals)),
        columns=["feature", "mean_abs_shap"]
        )

    if sort_desc:
        shap_mas_df.sort_values(
            by="mean_abs_shap",
            ascending=False,
            inplace=True)

    return shap_mas_df
