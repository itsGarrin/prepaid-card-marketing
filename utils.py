import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras import backend as K
import tensorflow as tf
from tqdm.notebook import tqdm


def calculate_vif(data: pd.DataFrame, vif_threshold: float = 5, correlation_threshold: float = 0.95,
                  variance_threshold: float = 1e-6) -> pd.DataFrame:
    """
    Iteratively calculate and remove variables with high Variance Inflation Factor (VIF) and high correlation.

    This function calculates the VIF for each variable in the provided DataFrame, removes variables with high correlation
    (greater than `correlation_threshold`), and removes variables with high VIF (greater than `vif_threshold`).
    It prints the names of removed variables and their VIF values once the process is complete.
    A progress bar shows the progress of removing high VIF and highly correlated features.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the variables to be evaluated.
    vif_threshold : float, optional
        The VIF threshold above which variables will be removed. Default is 5.
    correlation_threshold : float, optional
        The correlation threshold above which variables will be removed. Default is 0.95.
    variance_threshold : float, optional
        Minimum variance threshold below which columns are removed before VIF calculation to avoid divide-by-zero errors.
    n_jobs : int, optional
        The number of jobs to run in parallel for VIF calculation. Default is -1 (use all available cores).

    Returns:
    -------
    pd.DataFrame
        The DataFrame with variables having high VIF or high correlation removed.
    """
    removed_variables = []

    # Remove invariant features
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(data)
    data = data[data.columns[selector.get_support(indices=True)]]

    # Remove highly correlated variables
    corr_matrix = data.corr().abs()
    upper_triangle = np.triu(corr_matrix.values, 1)  # Upper triangle of the correlation matrix
    correlated_vars = set()

    # Iterate over the upper triangle to find pairs of highly correlated variables
    for i in range(len(upper_triangle)):
        for j in range(i + 1, len(upper_triangle[i])):
            if upper_triangle[i, j] > correlation_threshold:
                correlated_vars.add(corr_matrix.columns[j])

    # Remove highly correlated columns
    if correlated_vars:
        data = data.drop(columns=correlated_vars)
        print(f"Removed highly correlated variables: {list(correlated_vars)}")

    # Start the process of removing variables with high VIF
    with tqdm(total=len(data.columns), desc="Removing high VIF and correlated variables") as pbar:
        while True:
            # Calculate VIF using matrix operations after each removal
            corr_matrix = data.corr()  # Compute the correlation matrix
            inv_corr_matrix = np.linalg.inv(corr_matrix)  # Inverse of the correlation matrix

            # VIF is the diagonal of the inverse correlation matrix
            vif_values = np.diag(inv_corr_matrix)

            # Store VIF results in a DataFrame for easy access
            vif_df = pd.DataFrame({
                'variables': data.columns,
                'VIF': vif_values
            })

            # Find the variable with the highest VIF
            max_vif_row = vif_df.loc[vif_df['VIF'].idxmax()]
            max_vif = max_vif_row['VIF']
            removed_var = max_vif_row['variables']

            # If the highest VIF is above the threshold, remove the variable
            if max_vif > vif_threshold:
                data = data.drop(columns=removed_var)
                removed_variables.append((removed_var, max_vif))
                pbar.set_postfix({"Removed": removed_var, "VIF": max_vif})
                pbar.update(1)
                print(f"Removed variable: {removed_var} with VIF: {max_vif:.2f}")
            else:
                break

    # Print out the removed variables and their VIF values
    if removed_variables:
        print("Removed variables with high VIF:")
        for var, vif_value in removed_variables:
            print(f"{var}: {vif_value:.2f}")

    return data


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot the ROC curve for a given set of true labels and predicted probabilities.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    model_name : str, optional
        The name of the model (used for the plot title). Default is "Model".

    Returns:
    -------
    None
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--"
    )  # Diagonal line for random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {model_name}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def evaluate_models(
        models: list, predictions_base_train: list, predictions_base_test: list,
        predictions_hyper_train: list, predictions_hyper_test: list, X: pd.DataFrame, y_train: list, y_test: list,
        task: str = 'classification'
) -> pd.DataFrame:
    """
    Compare the performance of base models and models with hyperparameter tuning on both training and testing data.
    Returns a DataFrame with detailed metrics for each model.

    Parameters:
    ----------
    models : list
        List of model names.
    predictions_base_train : list
        List of predicted values from the base models on the training set.
    predictions_base_test : list
        List of predicted values from the base models on the testing set.
    predictions_hyper_train : list
        List of predicted values from hyperparameter-tuned models on the training set.
    predictions_hyper_test : list
        List of predicted values from hyperparameter-tuned models on the testing set.
    y_train : array-like
        Ground truth (correct) labels for the training set.
    y_test : array-like
        Ground truth (correct) labels for the testing set.
    task : str
        Type of evaluation ('classification' or 'regression').

    Returns:
    -------
    pd.DataFrame
        A DataFrame with detailed metrics for both base and hyperparameter-tuned models.
    """

    def compute_classification_metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "Positive Precision": precision_score(y_true, y_pred, pos_label=1),
            "Negative Precision": precision_score(y_true, y_pred, pos_label=0),
            "Positive Recall": recall_score(y_true, y_pred, pos_label=1),
            "Negative Recall": recall_score(y_true, y_pred, pos_label=0),
            "Positive F1 Score": f1_score(y_true, y_pred, pos_label=1),
            "Negative F1 Score": f1_score(y_true, y_pred, pos_label=0),
        }

    def compute_regression_metrics(y_true, y_pred):
        return {
            "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
            "Mean Squared Error": mean_squared_error(y_true, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R-squared": r2_score(y_true, y_pred),
            "Adjusted R-squared": 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (
                        len(y_true) - X.shape[1] - 1)
        }

    # Check for input validity
    if len(predictions_base_train) != len(models) or len(predictions_base_test) != len(models) or \
            len(predictions_hyper_train) != len(models) or len(predictions_hyper_test) != len(models):
        raise ValueError("The number of predictions must match the number of models.")

    if len(y_test) == 0 or len(y_train) == 0:
        raise ValueError("y_test and y_train cannot be empty.")

    # Compute metrics for both training and testing data
    all_metrics_train_base = {}
    all_metrics_test_base = {}
    all_metrics_train_hyper = {}
    all_metrics_test_hyper = {}

    for model, y_pred_base_train, y_pred_base_test, y_pred_hyper_train, y_pred_hyper_test in zip(
            models, predictions_base_train, predictions_base_test, predictions_hyper_train, predictions_hyper_test):

        if task == 'classification':
            all_metrics_train_base[model] = compute_classification_metrics(y_train, y_pred_base_train)
            all_metrics_test_base[model] = compute_classification_metrics(y_test, y_pred_base_test)
            all_metrics_train_hyper[model] = compute_classification_metrics(y_train, y_pred_hyper_train)
            all_metrics_test_hyper[model] = compute_classification_metrics(y_test, y_pred_hyper_test)
        elif task == 'regression':
            all_metrics_train_base[model] = compute_regression_metrics(y_train, y_pred_base_train)
            all_metrics_test_base[model] = compute_regression_metrics(y_test, y_pred_base_test)
            all_metrics_train_hyper[model] = compute_regression_metrics(y_train, y_pred_hyper_train)
            all_metrics_test_hyper[model] = compute_regression_metrics(y_test, y_pred_hyper_test)
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    # Initialize DataFrames for training and testing results
    metrics = list(all_metrics_train_base[models[0]].keys())

    results_train_base = pd.DataFrame(index=metrics, columns=models)
    results_test_base = pd.DataFrame(index=metrics, columns=models)
    results_train_hyper = pd.DataFrame(index=metrics, columns=models)
    results_test_hyper = pd.DataFrame(index=metrics, columns=models)

    # Fill DataFrames with metrics
    for model in models:
        for metric in metrics:
            results_train_base.loc[metric, model] = all_metrics_train_base[model][metric]
            results_test_base.loc[metric, model] = all_metrics_test_base[model][metric]
            results_train_hyper.loc[metric, model] = all_metrics_train_hyper[model][metric]
            results_test_hyper.loc[metric, model] = all_metrics_test_hyper[model][metric]

    # Label each DataFrame for concatenation
    results_train_base["Type"] = "Train - Base"
    results_test_base["Type"] = "Test - Base"
    results_train_hyper["Type"] = "Train - Hyperparameter Tuning"
    results_test_hyper["Type"] = "Test - Hyperparameter Tuning"

    # Combine all results
    results_combined = pd.concat([results_train_base, results_test_base, results_train_hyper, results_test_hyper])

    results_combined.reset_index(inplace=True)
    results_combined.rename(columns={"index": "Metric"}, inplace=True)
    results_combined.set_index(["Metric", "Type"], inplace=True)

    # Summarize the best models for each metric on test data
    summary_list = []
    for metric in metrics:
        base_values_test = results_combined.xs("Test - Base", level=1).loc[metric]
        hyper_values_test = results_combined.xs("Test - Hyperparameter Tuning", level=1).loc[metric]

        if task == 'regression' and metric in ["R-squared", "Adjusted R-squared"] or task == 'classification':
            best_value = max(base_values_test.max(), hyper_values_test.max())
            best_model = base_values_test.idxmax() if base_values_test.max() >= hyper_values_test.max() else hyper_values_test.idxmax()
        else:
            best_value = min(base_values_test.min(), hyper_values_test.min())
            best_model = base_values_test.idxmin() if base_values_test.min() <= hyper_values_test.min() else hyper_values_test.idxmin()

        summary_list.append({"Metric": metric, "Best Model": best_model, "Best Value": best_value})

    summary_df = pd.DataFrame(summary_list)
    summary_df.set_index("Metric", inplace=True)

    # Combine the combined results DataFrame with the summary DataFrame
    final_results = results_combined.join(summary_df, how="left", on=None, validate="many_to_many")

    return final_results


def plot_coefficients(ols_results: sm, highlight_vars=None, significance_level=0.05):
    """
    Plots the coefficients of an OLS model with p-values annotated.

    Parameters:
    ols_results: The fitted OLS model results from statsmodels.
    highlight_vars: List of variable names to highlight in the plot.
    significance_level: The p-value threshold for statistical significance.
    """

    # Create a DataFrame for coefficients and p-values
    coef_df = pd.DataFrame({
        'coefficients': ols_results.params.drop('const'),
        'pvalues': ols_results.pvalues.drop('const')
    })

    # Sort by absolute value of coefficients
    coef_df['abs_coeff'] = coef_df['coefficients'].abs()
    coef_df = coef_df.sort_values(by='abs_coeff')

    # Highlight significant coefficients
    significant = coef_df['pvalues'] < significance_level

    plt.figure(figsize=(10, 7))
    bars = plt.bar(coef_df.index, coef_df['coefficients'], color=['red' if sig else 'blue' for sig in significant])
    plt.axhline(0, color='k', linestyle='--')
    plt.title('OLS Coefficients with Significance Highlighted')

    # Annotate p-values on the bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{coef_df['pvalues'].iloc[i]:.3f}", ha='center', va='bottom')

    # Highlight specific variables
    if highlight_vars is not None:
        for i, bar in enumerate(bars):
            if coef_df.index[i] in highlight_vars:
                bar.set_color('orange')

    plt.xticks(rotation=45)
    plt.ylabel('Coefficient Value')
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

def plot_loss(history):
    """
    Plot the training and validation loss for a Keras model.

    Parameters:
    ----------
    history : keras.callbacks.History
        The history object returned by the model.fit() function.

    Returns:
    -------
    None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
