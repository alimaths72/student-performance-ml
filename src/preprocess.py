import pandas as pd

def prepare_features(df, for_training=True):
    df = df.copy()

    # Target
    if for_training:
        y = df["G3"]
        df = df.drop("G3", axis=1)
    else:
        y = None

    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Fill missing values
    df = df.fillna(0)

    return df, y
