from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (accuracy_score, auc, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import lightgbm as lgb
import matplotlib.pyplot as plt
import utils.modelutils as mu
import shap

model_ref = "experienced_any_public_wo"
balance_target = False
requested_balance = 0.5
save = True

# Read dataset
df_model = pd.read_csv("data/demog_optional_experienced_public_wo.csv")

# Counts
n_cases = len(df_model)
print(f"Model to receive dataset with {n_cases:,} cases")

tgt_vc = df_model.value_counts("target")
pos_pct = tgt_vc[1] / tgt_vc.sum()
print(f"Percentage of positive cases is {pos_pct:.1%}")

# Train test split
print("Creating train and test split of supplied data")
X = df_model.drop(["id", "target", "weight"],
                  axis=1)
y = df_model["target"]
# w = df_model["weight"]

(X_train, X_test,
 y_train, y_test
 #  ,w_train, w_test
 ) = train_test_split(X,
                      y,
                      #   w,
                      test_size=0.3,
                      random_state=0)

# Balance target?
if balance_target:
    try:
        us_scale = requested_balance
        print("Undersampling majority class")
        print(f"Requested target balance: {us_scale:.1%}")
        rus = RandomUnderSampler(random_state=1,
                                 sampling_strategy=us_scale)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    except ValueError as ve:
        print(ve)
        print("Continuing model build with native target " +
              f"balance: {pos_pct:.1%} " +
              f"(requested: {requested_balance:.1%})")

eval_metrics = ["binary_logloss", "auc"]
print("Initialising and fitting model")
model = lgb.LGBMClassifier(objective="binary")
model.fit(
        X_train,
        y_train,
        # sample_weight=w_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=eval_metrics,
        callbacks=[lgb.early_stopping(10)]
    )


# Diagnostics
print("Generating model diagnostics")
figs = {}

# Plot evaluation metrics
plot_ref = model_ref
for eval_met in eval_metrics:
    fig_e, ax_e = plt.subplots()
    lgb.plot_metric(model,
                    metric=eval_met,
                    title=f"{plot_ref} - Metric During Training",
                    ax=ax_e)
    fig_e.set_facecolor("white")
    plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_e.text(0.99, 0.01, plot_timestamp, ha="right",
               fontsize=8, alpha=0.3)
    if save:
        plt.savefig(f"results/eval_{eval_met}.png")
    figs[f"eval_{eval_met}"] = fig_e

# Plot feature importance
for imp_type in ["gain", "split"]:
    fig_i, ax_i = plt.subplots()
    lgb.plot_importance(
        model,
        importance_type=imp_type,
        max_num_features=10,
        xlabel=imp_type.title(),
        title=f"{plot_ref} - Feature Importance",
        ax=ax_i)
    fig_i.set_facecolor("white")
    plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig_i.text(0.99, 0.01, plot_timestamp, ha="right",
               fontsize=8, alpha=0.3)
    if save:
        plt.savefig(f"results/feature_imp_{imp_type}.png")
    figs[f"feature_imp_{imp_type}"] = fig_i


# Evaluate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred, y_prob = y_pred, y_prob

accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
auc_roc = roc_auc_score(y_test, y_prob)
auc_pr = auc(recall, precision)
f1 = f1_score(y_test, y_pred)
prob_min = y_prob.min()
prob_max = y_prob.max()


# Confusion matrices
confusion_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix, display_labels=model.classes_)
fig_cm, ax_cm = plt.subplots()
disp.plot(values_format=",",
          colorbar=False,
          cmap="Blues",
          ax=ax_cm).ax_.set_title(f"{plot_ref} - " +
                                  "Confusion Matrix " +
                                  f"(F1: {f1:.3f})")
fig_cm.set_facecolor("white")
plot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
fig_cm.text(0.65, 0.01, plot_timestamp, fontsize=8, alpha=0.3)
if save:
    plt.savefig("results/confusion_matrix.png")
figs["confusion_matrix"] = fig_cm

confusion_matrix_pc = (confusion_matrix /
                       confusion_matrix.sum())
disp_pc = ConfusionMatrixDisplay(confusion_matrix_pc,
                                 display_labels=model.classes_)

fig_cm_pc, ax_cm_pc = plt.subplots()
disp_pc.plot(values_format=".1%",
             colorbar=False,
             cmap="Blues",
             ax=ax_cm_pc).ax_.set_title(f"{plot_ref}" +
                                        " - Confusion Matrix " +
                                        f"(F1: {f1:.3f})")
fig_cm_pc.set_facecolor("white")
fig_cm_pc.text(0.65, 0.01, plot_timestamp, fontsize=8, alpha=0.3)
if save:
    plt.savefig("results/confusion_matrix_pc.png")
figs["confusion_matrix_pc"] = fig_cm_pc

fig_roc = mu.plot_roc(fpr, tpr,
                      auc_value=auc_roc,
                      model_name=plot_ref)
if save:
    plt.savefig("results/roc.png")
figs["roc"] = fig_roc
fig_pr = mu.plot_pr(recall, precision,
                    auc_value=auc_pr,
                    model_name=plot_ref)
if save:
    plt.savefig("results/precision_recall.png")
figs["precision_recall"] = fig_pr

score_dist = mu.plot_score_dist(list(y_prob), plot_ref)
if save:
    plt.savefig("results/score_dist.png")
figs["score_dist"] = score_dist


# SHAP
print("Generating shapley values")

plot_ref = model_ref
explainer = shap.TreeExplainer(model)
X_shap = X_test[model.feature_name_]
shap_values = explainer.shap_values(X_shap)[1]

fig_shap = plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
fig_shap.set_facecolor("white")
fig_shap.suptitle(f"{plot_ref} - SHAP Summary Plot")
if save:
    plt.savefig("results/shap_summary.png")
figs["shap_summary"] = fig_shap

shap_mas_df = mu.mean_abs_shap(shap_values, X_test.columns)
shap_top_n = shap_mas_df["feature"][0:20]

counter = 0
fig_shap_dp = {}

for f in shap_top_n:
    counter += 1
    fig_dp, ax_dp = plt.subplots()
    shap.dependence_plot(
        f,
        shap_values,
        X_shap,
        interaction_index=None,
        alpha=0.5,
        ax=ax_dp,
        title=f"{plot_ref} - F{counter}: {f}"
        )
    fig_dp.set_facecolor("white")
    if save:
        plt.savefig(f"results/shap_partial_f{counter}.png")
    fig_shap_dp[f"f{counter}"] = fig_dp

figs["shap_dependence_plots"] = fig_shap_dp
