import pandas as pd
import numpy as np

from math import ceil 


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, log_loss, confusion_matrix, precision_score, recall_score
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

from typing import Any, Tuple

# from scipy import stats
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import SMOTENC


# below are various supporting functions
def plot_na_cats(df: pd.DataFrame) -> None:
    """Plot nan categories."""

    nvars = len(df.columns)
    ncols = 5
    nrows = ceil(nvars / ncols) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        df[col].plot(kind='bar', ax=axes[i], title=col)
        axes[i].set_ylabel('Value')
        axes[i].set_xlabel('Index')

    # remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def generate_datasets(df: pd.DataFrame, test_size: float=0.2, random_state: int=23) -> tuple:
    """Generate training sets."""

    df = df.astype("Int32")

    df_fulltrain, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    val_size = round(df_test.shape[0] / df_fulltrain.shape[0], 2)
    df_train, df_val = train_test_split(df_fulltrain, test_size=val_size, random_state=random_state)

    print(f"""From {df.shape[0]}, rows generated:
    df_fulltrain: {df_fulltrain.shape[0]}
    df_train: {df_train.shape[0]}
    df_val: {df_val.shape[0]}
    df_test: {df_test.shape[0]}""")

    return df_fulltrain, df_train, df_val, df_test

def plot_dist(df: pd.DataFrame, columns: list) -> None:  
    """Plot variable distributions."""  

    nvars = len(df.columns)
    ncols = 5
    nrows = ceil(nvars / ncols) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        title_color = ["red" if col in columns else "black"][0]
        ax = axes[i]
        ax.hist(df[col])
        ax.set_title(col, color=title_color)
        ax.set_ylabel('Value')
        ax.set_xlabel('Frequency')

    # remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def recompute_na(df: pd.DataFrame, imputation_guide: pd.DataFrame) -> pd.DataFrame:
    """Update imputation guide based on missing data."""

    tot_rows = df.shape[0]
    na_vals = df.isna().sum()
    na_percs = round(((na_vals / tot_rows) * 100), 2)

    df_na = {}
    for col, vals in zip(na_vals.index, zip(na_vals, na_percs)):
        count = vals[0]
        perc = vals[1]
        df_na[col] = {"na_count": count, "na_percentage": perc}

    df_na = pd.DataFrame(df_na)
    df_na = df_na.T.sort_values(by="na_percentage", ascending=False)
    df_na["imputation_technique"] = np.where(df_na.na_percentage == 0, "-", df_na.index.to_series().map(imputation_guide))
    
    return df_na

def verify_cardinality(df_train: pd.DataFrame, df_other: pd.DataFrame) -> dict:
    """Check if there are any missing classes between the datasets."""

    check = {}

    for col in df_train.columns:
        result = len(df_train[col].unique()) - len(df_other[col].unique())

        if result != 0:
            check[col] = result
    
    check = dict(sorted(check.items(), key=lambda item: item[1], reverse=True))

    return check

def plot_labels(df: pd.DataFrame) -> None:
    """
    Plot the frequency of labels in a df and flag labels representing
    less than 5% of the data.
    """
    
    nvars = len(df.columns)
    ncols = 5
    nrows = ceil(nvars / ncols)
    tot_rows = df.shape[0]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        vals = pd.Series(df[col].value_counts() / tot_rows)
        ax = axes[i]
        vals.sort_values(ascending=False).plot.bar(ax=ax)
        # target labels for which the frequency represents less than 5% of the data 
        ax.axhline(y=0.05, color="red")
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Labels')
        ax.set_title(col)

    for j in range (i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def get_objname(obj: Any) -> str:
    """Retrieve the name of an object (for formating purposes)."""

    obj_name =[v for v in globals() if globals()[v] is obj][0]
    
    return obj_name

def get_country_weights(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Assign a weight to each country such as that n=1000."""

    df = df.copy()

    col_name = col + "_weights"
    weights = dict(1000 / df[col].value_counts())
    df[col_name] = df[col].map(weights)

    return df

def get_rare_label_weights(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """
    Assign weights to column values such as that the product between the weights and
    the frequency of values is rescaled to represent proportions of data going from 90% to 10%.
    For instance, if one label of a variable represents 3% of the data it's weight will bring it
    to 10%. The whole label distribution of the variables will be rescaled to a 90%-10% range.
    """

    df = df.copy()
    upper_bound = 0.9
    lower_bound = 0.1

    for col in col_list:
        col_name = col + "_weights"
        # get row count for the target column
        tot_rows = len(df[col].dropna())
        # get label frequency
        series = df[col].value_counts() / tot_rows
        # get minimum
        minimum = series.min()
        # get maximum
        maximum = series.max()
        # normalize distribution
        normalized_series = (series - minimum) / (maximum - minimum)
        # adjust normalization
        adjusted_series = normalized_series * (upper_bound - lower_bound) + lower_bound
        # get the weights
        weights = dict(adjusted_series / series)
        # map the weights to the levels
        df[col_name] = df[col].map(weights)
        
    return df

def format_weights(df: pd.DataFrame, targets: list, original_weights: dict) -> np.array:
    """
    Iterate over the df, combine the weights and return a
    re-normalized array ranging from 1 to 10.
    
    params: 
        targets=columns to be weighted
        original_weights: dict with pairs index-weights
    """

    # get country weights
    df = get_country_weights(df, "country")
    # get normalized weights
    df = get_rare_label_weights(df, targets)
    # get original weights 
    df["original_weights"] = df.index.map(original_weights)

    # get weighted columns
    wcols = [col for col in df.columns if "weights" in col]
    # multiply weights together
    df["weights"] = df[wcols].prod(axis=1)
    # control weight magnitude by normalizing them
    min = df.weights.min()
    max = df.weights.max()
    ubound = 10
    lbound= 1
    df["norm_weights"] = (df.weights - min) / (max - min)
    df["adj_weights"] = df.norm_weights * (ubound - lbound) + lbound

    weights = df.adj_weights.values

    return weights

def plot_xgb_output(output) -> pd.DataFrame:
    results = []
    columns = ["num_iter", "train_auc", "val_auc"]

    for line in output.stdout.strip().split("\n"):
        it_line, train_line, val_line = line.split("\t")

        it = int(it_line.strip("[]"))
        train = float(train_line.split(":")[1])
        val = float(val_line.split(":")[1])

        results.append((it, train, val))

    df_results = pd.DataFrame(results, columns=columns)

    # plot scores
    plt.plot(df_results.num_iter, df_results.train_auc, label="train")
    plt.plot(df_results.num_iter, df_results.val_auc, label="val")
    plt.legend()
    plt.show()

    return df_results

def get_youden_index(ytrue: np.array, ypred: np.array) -> Tuple[float, float, int, int]:
    """Find the best youden Index between ytrue and ypred and return its accuracy, 
    precision, sensitivity & confusion matrix"""

    # get estimates 
    fpr, tpr, thresholds = roc_curve(ytrue, ypred)

    # get threshold minimising TPR - FPR
    threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
    # youden_threshold = f"Best Balanced Threshold (maximising TPR - FPR): {threshold: .2f}"
    # turn predictions in to discrete label
    ypred = (ypred >= threshold).astype(int)
    # get accuracy 
    accuracy = accuracy_score(ytrue, ypred)
    # get precision 
    precision = round(precision_score(ytrue, ypred) * 100)
    # get recall
    sensitivity = round(recall_score(ytrue, ypred) * 100)
    # get confusion matrix
    cm = confusion_matrix(ytrue, ypred)

    return threshold, accuracy, precision, sensitivity, cm

def get_best_acc_estimates(ytrue: np.array, ypred: np.array) -> Tuple[float, float, int, int]:
    """Find the threshold with the best accuracy between ytrue and ypred and return 
    its accuracy, precision, sensitivity & confusion matrix"""

    # get estimates 
    fpr, tpr, thresholds = roc_curve(ytrue, ypred)
    best_accuracy = 0
    best_threshold = 0

    # find threshold with highest accuracy
    for threshold in thresholds:
        n_ypred = (ypred >= threshold).astype(int)
        accuracy = accuracy_score(ytrue, n_ypred)

        if accuracy > best_accuracy:
            best_accuracy = round(accuracy, 3)
            best_threshold = round(threshold, 3)
            # save best prediction
            chosen_ypred = n_ypred
    
    # get precision
    precision = round(precision_score(ytrue, chosen_ypred) * 100)
    # get recall
    sensitivity = round(recall_score(ytrue, chosen_ypred) * 100)
    # get confusion matrix
    cm = confusion_matrix(ytrue, chosen_ypred)


    return best_threshold, best_accuracy, precision, sensitivity, cm

def optimize_cm(ytrue: np.array, ypred: np.array, threshold: float=None) -> None:
    """
    Finds the best classfication thresholds by separatly minimizing TPR - FPR and 
    maximising accuracy. Also returns the related estimates.
    """

    if not threshold:
        # get classification estimates based on minimization of TPR - FPR
        youden_threshold, youden_accuracy, youden_precision, youden_sensitivity, youden_cm = get_youden_index(ytrue, ypred)
        # get classfication estimates based on the threshold with the highest accuracy
        acc_threshold, acc_accuracy, acc_precision, acc_sensitivity, acc_cm = get_best_acc_estimates(ytrue, ypred)

    # plot cm
    df_index = ["highest_balance", "highest_accuracy"]
    df_columns = ["threshold", "accuracy_perc", "precision_perc", "sensitivity_perc"]
    data = [(youden_threshold, youden_accuracy, youden_precision, youden_sensitivity), (acc_threshold, acc_accuracy, acc_precision, acc_sensitivity)]
    cm_list = [youden_cm, acc_cm]

    # plot cms
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    for i, cm in enumerate(cm_list):
        ax = axes[i]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Eng", "Pred Dis"], 
                                                            yticklabels=["True Eng", "True Dis"], ax=ax)

        ax.set_title(f"Political Engagement Matrix ({df_index[i]})")
    
    plt.show()

    df = pd.DataFrame(data, index=df_index, columns=df_columns)
    display(df)


def get_cm(ytrue: np.array, ypred: np.array, threshold: float=0.5) -> None:
    """Plot the confusion matrix between ytrue and ypred and 
    return estimates for the given threshold."""

    # turn predictions to discrete label
    ypred = (ypred >= threshold).astype(int)
    # get accuracy 
    acc = accuracy_score(ytrue, ypred)
    # get precision 
    precision = round(precision_score(ytrue, ypred) * 100)
    # get recall
    sensitivity = round(recall_score(ytrue, ypred) * 100)
    # get confusion matrix
    cm = confusion_matrix(ytrue, ypred)

    # plot cm
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pred Eng", "Pred Dis"], 
                                                       yticklabels=["True Eng", "True Dis"])

    plt.title('Confusion Matrix on Political Engagement Level')
    plt.show()

    print(f"Estimates for threshold: {threshold}")
    print(f"Model Accuracy: {acc: .2f}%")
    print(f"Correctly predicted political engagement levels (precison) : {precision}%")
    print(f"Overall proportion of correctly identified profiles (sensitivity) : {sensitivity}%")

def wrangle_tuning_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    sort_logic = {"val_auc": False,     # sort val_auc from highest to lowest 
                  "val_loss": True,     # sort val_loss from lowest to highest
                  "auc_diff": True,     # sort auc_diff from lowest to highest
                  "loss_diff": False}   # sort loss_diff from highest to lowers (negative)
    
    df = df.copy()
    
    # let's filter under/overfitting models
    df = df[(df.train_auc > 0.5) & (df.train_auc <= 0.9)]

    # get estimate differences
    df["auc_diff"] = df.train_auc - df.val_auc
    df["loss_diff"] = df.train_loss - df.val_loss 

    col_prefix = "top_10_"

    # ideally, we would opt for a model with a high accuracy, small losss and a small auc and log loss difference
    # let's flag the 10 best models on every estimator
    for estim, asc in sort_logic.items():
        # first 10 best models get 1 point
        df[col_prefix + estim] = np.where(df.index.isin(df.sort_values(by=estim, ascending=asc).head(10).index), 1, 0)

    # get an aggregate score for best models
    top_cols = [col for col in df if "top" in col]
    df["top_models"] = df[top_cols].sum(axis=1)

    # save best models for comparison
    top_models = df[df.top_models > 0].sort_values(by="top_models", ascending=False).head(20)

    # return wrangled tuning results and top 10 models df
    return df, top_models

def get_xgboost_metric(model: xgb.core.Booster, metric: str, features: list) -> None:
    """Get original features from dummy features and return target metrics 
    for xgboost model (feature importance (weight), gain, cover, total_gain, total_cover)."""

    # get metric
    metric_score = model.get_score(importance_type=metric)
    # split model feature and metric score
    metric_df = pd.DataFrame({"feature": list(metric_score.keys()), metric: list(metric_score.values())})
    # get original columns
    
    # map original columns
    metric_df["original_feature"] = metric_df["feature"].apply(lambda v: v.split("=")[0] if "=" in v else v)
    # group by importance score
    agg_metric = metric_df.groupby("original_feature")[metric].mean().reset_index()
    agg_metric = agg_metric.sort_values(by=metric, ascending=True)

    # plot
    fig, ax = plt.subplots(figsize=(15, 10))  # Adjust figure size accordingly
    ax.barh(agg_metric["original_feature"], agg_metric[metric])
    ax.set_title(f"{metric} by Variable")
    plt.show()