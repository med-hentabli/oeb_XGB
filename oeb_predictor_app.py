# -*- coding: utf-8 -*-
"""
OEB Prediction Pro - XGBoost Only Version
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from urllib.parse import quote
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.special import softmax
import os
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(page_title="OEB Prediction Pro", layout="wide", page_icon="🔬")

# --- CONSTANTS ---
try:
    DESC_NAMES = [desc[0] for desc in Descriptors._descList]
except AttributeError:
    st.warning("Could not dynamically load RDKit descriptor names. Using a predefined list might be necessary if errors occur.")
    DESC_NAMES = []

OEB_DESCRIPTIONS = {
    0: "No exposure limits: Minimal or no systemic toxicity.",
    1: "OEB 1: Low hazard (OEL: 1000 - 5000 µg/m³)",
    2: "OEB 2: Moderate hazard (OEL: 100 - 1000 µg/m³)",
    3: "OEB 3: High hazard (OEL: 10 - 100 µg/m³)",
    4: "OEB 4: Very high hazard (OEL: 1 - 10 µg/m³)",
    5: "OEB 5: Extremely high hazard (OEL: < 1 µg/m³)",
    6: "OEB 6: Extremely potent (OEL: < 0.1 µg/m³)"
}

MODEL_NAME = "XGBoost"
DEFAULT_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
MODEL_DIR = "models"
CNN_MODEL_NAME = "cnn_model_tf213_compatiblev2"

def get_model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, MODEL_DIR, filename)

@st.cache_resource
def load_models_and_scalers():
    scalers = {}
    xgb_model = None
    cnn_model = None

    try:
        scalers = {
            "desc": joblib.load(get_model_path("scaler_descriptors.pkl")),
            "cnn_input": joblib.load(get_model_path("scaler_features_cnn.pkl")),
            "cnn_output": joblib.load(get_model_path("scaler_features_cnn_output.pkl"))
        }

        xgb_model = joblib.load(get_model_path(f"model_{MODEL_NAME}.pkl"))

        imported = tf.saved_model.load(get_model_path(CNN_MODEL_NAME))
        cnn_model = imported.signatures["serving_default"]

    except Exception as e:
        st.error(f"Model loading error: {e}")

    return cnn_model, scalers, xgb_model

def compute_cnn_ready_features(smiles, scalers, cnn_model):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        desc_calc = MolecularDescriptorCalculator(DESC_NAMES)
        descriptors = np.array(desc_calc.CalcDescriptors(mol))

        padded_desc = np.zeros(1024)
        padded_desc[:min(len(descriptors), 1024)] = descriptors[:min(len(descriptors), 1024)]
        norm_desc = scalers["desc"].transform([padded_desc])[0]

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_as_numpy_array = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, fp_as_numpy_array)

        combined_features = np.stack((norm_desc, fp_as_numpy_array), axis=-1)
        cnn_input_image = combined_features.reshape(32, 32, 2)
        norm_input_flat = scalers["cnn_input"].transform(cnn_input_image.reshape(1, -1))
        norm_input_reshaped = norm_input_flat.reshape(1, 32, 32, 2)

        input_tensor = tf.convert_to_tensor(norm_input_reshaped, dtype=tf.float32)
        output = cnn_model(input_tensor)
        features = output['output_0'].numpy() if 'output_0' in output else list(output.values())[0].numpy()

        cnn_features_scaled = scalers["cnn_output"].transform(features)
        return cnn_features_scaled
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_pubchem_data(compound_name):
    try:
        encoded_name = quote(compound_name)
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
        res_cid = requests.get(cid_url, timeout=10)
        res_cid.raise_for_status()
        cid = res_cid.json().get("IdentifierList", {}).get("CID", [None])[0]

        if cid:
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
            res_smiles = requests.get(smiles_url, timeout=10)
            res_smiles.raise_for_status()
            smiles = res_smiles.json().get("PropertyTable", {}).get("Properties", [{}])[0].get("CanonicalSMILES")
            pubchem_page_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            return pubchem_page_url, smiles
    except:
        pass
    return None, None

def normalize_probabilities(probs, target_length):
    normalized = np.zeros(target_length)
    common_len = min(len(probs), target_length)
    normalized[:common_len] = probs[:common_len]
    return normalized / np.sum(normalized) if np.sum(normalized) > 0 else np.full(target_length, 1/target_length)

def main():
    st.title("OEB Prediction Pro 🔬")
    st.markdown("Predict Occupational Exposure Bands for chemical compounds using XGBoost and CNN features.")

    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input = DEFAULT_SMILES

    cnn_model, scalers, xgb_model = load_models_and_scalers()
    if cnn_model is None or not scalers or xgb_model is None:
        st.error("Cannot proceed due to loading errors.")
        return

    st.sidebar.header("⚙️ Input Options")

    pubchem_name = st.sidebar.text_input("Compound name (PubChem)")
    if pubchem_name:
        pubchem_url, smiles_found = get_pubchem_data(pubchem_name)
        if smiles_found:
            st.sidebar.success(f"Found SMILES: {smiles_found}")
            if st.sidebar.button("Use this SMILES"):
                st.session_state.smiles_input = smiles_found

    smiles = st.text_input("Enter SMILES", st.session_state.smiles_input)

    if st.button("🚀 Predict OEB"):
        features = compute_cnn_ready_features(smiles, scalers, cnn_model)
        if features is None:
            st.error("Could not compute features.")
            return

        if hasattr(xgb_model, "predict_proba"):
            probs = xgb_model.predict_proba(features)[0]
        else:
            scores = xgb_model.decision_function(features)
            probs = softmax(scores, axis=1)[0] if scores.ndim > 1 else softmax([scores])[0]

        probs = normalize_probabilities(probs, len(OEB_DESCRIPTIONS))
        pred_class = int(np.argmax(probs))

        st.success(f"Predicted OEB: **{pred_class}**")
        st.markdown(f"**{OEB_DESCRIPTIONS.get(pred_class, 'Unknown')}**")

        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            "OEB": list(OEB_DESCRIPTIONS.keys()),
            "Probability": probs
        }).set_index("OEB")

        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}).bar(color="#5fba7d"))

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Molecular Properties")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mol Weight", f"{Descriptors.MolWt(mol):.2f}")
            with col2:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                st.metric("Fingerprint Bits", f"{len(fp.GetOnBits())}")

            if pubchem_name and pubchem_url:
                st.markdown(f"[View on PubChem ↗]({pubchem_url})")

if __name__ == "__main__":
    main()
