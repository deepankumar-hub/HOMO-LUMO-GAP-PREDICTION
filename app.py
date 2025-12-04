import streamlit as st
import requests
import joblib
import deepchem as dc
from rdkit import Chem
import numpy as np
from tensorflow import keras

MODEL_PATH = "rf_fp_model.pkl"

FEATURIZER_PATH = "fp_featurizer.pkl"

model = joblib.load(MODEL_PATH)
featurizer = joblib.load(FEATURIZER_PATH)

def iupac_to_smiles(iupac_name):
    url = f"https://cactus.nci.nih.gov/chemical/structure/{iupac_name}/smiles"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text.strip()
    return None


def featurize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = featurizer.featurize([smiles])
        return features

    except:
        return None


def predict_gap(features):
    X = np.asarray(features)
    if hasattr(model, 'model') and hasattr(model.model, 'predict'):
        try:
            pred = model.model.predict(X)
            return float(np.ravel(pred)[0])
        except Exception:
            ds = dc.data.NumpyDataset(X)
            pred = model.predict(ds)
            return float(np.ravel(pred)[0])

    if hasattr(model, 'predict'):
        pred = model.predict(X)
        return float(np.ravel(pred)[0])
    ds = dc.data.NumpyDataset(X)
    pred = model.predict(ds)
    return float(np.ravel(pred)[0])


st.title("HOMO–LUMO Gap Predictor")

iupac_input = st.text_input("Enter IUPAC name", placeholder="benzene, phenol, naphthalene...")

if st.button("Predict"):
    if not iupac_input.strip():
        st.error("Please enter a valid IUPAC name.")
    else:
        st.info("Converting IUPAC → SMILES...")
        smiles = iupac_to_smiles(iupac_input)

        if smiles is None:
            st.error("SMILES not found for this IUPAC name.")
        else:
            st.success(f"SMILES: {smiles}")

            st.info("Featurizing molecule...")
            features = featurize_smiles(smiles)

            if features is None:
                st.error("Error featurizing SMILES.")
            else:
                st.info("Running ML model...")
                gap = predict_gap(features)

                st.success(f"Predicted HOMO–LUMO gap: {gap:.4f} eV")
