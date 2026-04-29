import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data( data, categorical_features=[], label=None, encoder=None, lb=None):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Removed training as input... enter None for encoder to retrain categorical features and label binarizer. Enter trained encoder to re-use the one-hot encoder and label binarizer.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    #if no label given data is only features, otherwise separate features and labels
    if label == None:
        y = np.array([])
        X = data
    else:
        y = data[label]
        X = data.drop([label], axis=1)


    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
    else:
        X_categorical = encoder.transform(X_categorical)

    if label is not None:
        if lb is None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y.values).ravel()
        else:
            y = lb.transform(y.values).ravel()

    X = np.concatenate([X_continuous.values, X_categorical], axis=1)
    return X, y, encoder, lb
