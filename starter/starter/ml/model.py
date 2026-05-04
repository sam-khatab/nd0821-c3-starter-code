from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_rc_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)

def slice_metrics(data, model, feature, value):
    """
    Computes metrics for a specific slice of the data.

    Inputs
    ------
    data : dataframe containing the features and labels

    feature : str
        Feature name to slice on.
    value : any
        Value of the feature to slice on.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    data = data[data[feature]==value]

    X = data.drop("salary", axis=1)
    y = data["salary"]
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    with open("slice_output.txt", "a") as fp:
        fp.write(f"Feature: {feature}, Value: {value}\n") 
        fp.write(f"    Precision: {precision}, Recall: {recall}, F1: {fbeta}\n\n")
    