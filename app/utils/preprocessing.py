import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

def load_csv(file_path, col_names=None):
    try:
        if "housing" in file_path.lower():
            data = pd.read_csv(file_path, names=col_names, delim_whitespace=True)
        else:
            data = pd.read_csv(file_path, names=col_names)
        return data
    except Exception as e:
        print("Erreur lors du chargement du fichier :", e)
        return None

def split_features_target_df(df, target_col):
    """Renvoie X, y à partir d’un DataFrame et de la colonne cible."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def split_train_test(X, y, test_size=0.3, seed=11):
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def get_scaler(name: str):
    """Retourne un scaler sklearn selon le nom."""
    if name == "StandardScaler":
        return StandardScaler()
    elif name == "Normalizer":
        return Normalizer()
    elif name == "MinMaxScaler":
        return MinMaxScaler()
    return None 

def fit_transform_scaler(scaler, X_train, X_test):
    if scaler is None:
        return X_train, X_test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
