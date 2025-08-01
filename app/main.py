import streamlit as st
import os
import pandas as pd
import numpy as np

from utils.preprocessing import (
    load_csv, split_features_target_df, split_train_test,
    get_scaler, fit_transform_scaler
)
from utils.models import get_classification_models, get_regression_model
from utils.metrics import evaluate_classification, evaluate_regression

st.set_page_config(page_title="Apprentissage Automatique", layout="wide")
st.title("Application d'Apprentissage Automatique (cours)")

if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.task_type = None
    st.session_state.model_name = None
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.features_cols = None
    st.session_state.target_col = None


tab1, tab2, tab3 = st.tabs([" Upload & Exploration", " Entraînement ML", " Prédiction"])


with tab1:
    st.subheader("Télécharger un CSV")

    uploaded_file = st.file_uploader("Choisissez un fichier (Pima / Housing / autre)", type=["csv"])
    if uploaded_file is not None:
        
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(" Fichier enregistré dans /data")

        if "pima" in uploaded_file.name.lower():
            col_names = ['grossesse','glucos','PresArt','Epaiss','Insuline','IMC','Pedigre','Age','class']
        elif "housing" in uploaded_file.name.lower():
            col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAXE','PTRATIO','B','LSTAT','MEDV']
        else:
            col_names = None

        df = load_csv(file_path, col_names=col_names)
        if df is not None:
            st.write("Aperçu :")
            st.dataframe(df.head())
            st.write("Shape :", df.shape)
            st.session_state.df = df  


with tab2:
    st.subheader("Entraînement des modèles")

    if "df" not in st.session_state:
        st.info(" Uploade d'abord un fichier dans l'onglet **Upload & Exploration**.")
    else:
        df = st.session_state.df

        task_type = st.radio("Type de tâche", ["Classification", "Régression"], key="task_type_radio")

        target_col = st.selectbox("Colonne cible (output)", df.columns, key="target_col_select")

        scaler_name = st.selectbox(
            "Normalisation (optionnel)",
            ["Aucune", "StandardScaler", "Normalizer", "MinMaxScaler"],
            index=0
        )

        if task_type == "Classification":
            models = get_classification_models()
            model_name = st.selectbox("Modèle", list(models.keys()))
            model = models[model_name]
        else:
            model_name = "LinearRegression"
            model = get_regression_model()

        if st.button(" Entraîner"):
            X, y = split_features_target_df(df, target_col)
            features_cols = X.columns.tolist()

            x_train, x_test, y_train, y_test = split_train_test(X.values, y.values)

            scaler = get_scaler(scaler_name if scaler_name != "Aucune" else None)
            x_train_scaled, x_test_scaled = fit_transform_scaler(scaler, x_train, x_test)

            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)

            if task_type == "Classification":
                scores = evaluate_classification(y_test, y_pred)
            else:
                scores = evaluate_regression(y_test, y_pred)

            st.success(" Entraînement terminé !")
            st.markdown(f"**Modèle :** {model_name}")
            st.json(scores)

            st.session_state.trained = True
            st.session_state.task_type = task_type
            st.session_state.model_name = model_name
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features_cols = features_cols
            st.session_state.target_col = target_col


with tab3:
    st.subheader("Prédictions sur de nouvelles données")

    if not st.session_state.trained:
        st.info("Entraîne d'abord un modèle dans l’onglet **Entraînement ML**.")
    else:
        task_type = st.session_state.task_type
        model = st.session_state.model
        scaler = st.session_state.scaler
        features_cols = st.session_state.features_cols

        st.write("**Modèle entraîné :**", st.session_state.model_name)
        st.write("**Type de tâche :**", task_type)

        mode = st.radio("Comment fournir les données ?", ["Saisie manuelle", "Uploader un CSV (mêmes colonnes X)"])

        if mode == "Saisie manuelle":
            inputs = {}
            for col in features_cols:
                inputs[col] = st.number_input(f"{col}", value=0.0)
            if st.button(" Prédire"):
                X_new = pd.DataFrame([inputs])[features_cols].values
                if scaler is not None:
                    X_new = scaler.transform(X_new)
                y_pred = model.predict(X_new)
                st.success(f"**Prédiction :** {y_pred[0]}")

        else:
            pred_file = st.file_uploader("Uploader un CSV contenant uniquement les features (sans la colonne cible)", type=["csv"])
            if pred_file is not None:
                df_new = pd.read_csv(pred_file)
                st.write("Aperçu des données à prédire :")
                st.dataframe(df_new.head())

                try:
                    X_new = df_new[features_cols].values
                except KeyError:
                    st.error("Les colonnes du CSV ne correspondent pas aux features utilisées pour l'entraînement.")
                    st.stop()

                if st.button("Prédire (batch)"):
                    if scaler is not None:
                        X_new = scaler.transform(X_new)
                    y_pred = model.predict(X_new)
                    st.write("**Prédictions :**")
                    st.dataframe(pd.DataFrame({"prediction": y_pred}))
