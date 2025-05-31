import streamlit as st
import pandas as pd
import numpy as np
import os
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.models import load_model

# === Ścieżki ===
MODEL_DIR = "models"
TEST_DATA_PATH = "złączone_dane.xlsx"

st.set_page_config(layout="wide", page_title="Dashboard ML - Predykcja")
st.title("📊 Interaktywny dashboard do testowania modeli ML")

model_choice = st.selectbox("Wybierz model", ["MLP", "Random Forest", "TensorFlow", "PyTorch"])

# === Wczytaj dane ===
@st.cache_data
def load_test_data():
    df = pd.read_excel(TEST_DATA_PATH)
    df = df.drop(columns=[col for col in df.columns if any(x in col for x in ['3_p', '4_p', '5_p'])], errors='ignore')
    if 'image_id' in df.columns:
        df = df.drop('image_id', axis=1)
    return df.reset_index(drop=True)

df = load_test_data()

# Obsługa przypadku z brakiem danych
if df.empty:
    st.error("Brak danych do analizy. Sprawdź plik testowy.")
    st.stop()

# Oddzielenie X i y (jeśli y istnieje)
if 'label' in df.columns:
    y = df['label']
    X = df.drop('label', axis=1)
else:
    X = df
    y = None

# Slider tylko gdy więcej niż jedna próbka
if len(X) > 1:
    sample_idx = st.slider("Wybierz indeks próbki", 0, len(X) - 1, 0)
else:
    sample_idx = 0
    st.warning("Znaleziono tylko jedną próbkę.")

# === Preprocessing ===
def preprocess_features(X: pd.DataFrame, n_components: int = 1) -> pd.DataFrame:
    lm_0_cols = [col for col in X.columns if col.startswith('0_point_lm_')]
    lm_1_cols = [col for col in X.columns if col.startswith('1_point_lm_')]
    lm_2_cols = [col for col in X.columns if col.startswith('2_point_lm_')]
    vec_cols = [col for col in X.columns if '_vec_' in col]

    def pca_transform(cols, prefix):
        if not cols:
            return pd.DataFrame(index=X.index)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[cols])
        pca = PCA(n_components=min(n_components, len(cols)))
        X_pca = pca.fit_transform(X_scaled)
        return pd.DataFrame(X_pca, columns=[f'{prefix}_pca_{i}' for i in range(X_pca.shape[1])], index=X.index)

    pca_lm_0 = pca_transform(lm_0_cols, '0')
    pca_lm_1 = pca_transform(lm_1_cols, '1')
    pca_lm_2 = pca_transform(lm_2_cols, '2')
    vec_features = X[vec_cols].reset_index(drop=True)
    return pd.concat([pca_lm_0, pca_lm_1, pca_lm_2, vec_features], axis=1)

X_test = preprocess_features(X, n_components=1)

# === Predykcja ===
def predict_mlp(X):
    model = load(os.path.join(MODEL_DIR, "mlp_best_model.pkl"))
    le = load(os.path.join(MODEL_DIR, "mlp_label_encoder.pkl"))
    proba = model.predict_proba([X.iloc[sample_idx]])
    classes = le.inverse_transform(np.arange(len(proba[0])))
    return dict(zip(classes, np.round(proba[0], 4)))

def predict_rf(X):
    model = load(os.path.join(MODEL_DIR, "Randomforest_best.pkl"))
    proba = model.predict_proba([X.iloc[sample_idx]])[0]
    classes = model.classes_ if hasattr(model, "classes_") else [str(i) for i in range(len(proba))]
    return dict(zip(classes, np.round(proba, 4)))

def predict_tf(X):
    model = load_model(os.path.join(MODEL_DIR, "tf_ssn_model.h5"))
    classes = np.load(os.path.join(MODEL_DIR, "label_encoder_classes.npy"), allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # W praktyce: użyj trenowanego skalera!
    proba = model.predict(X_scaled, verbose=0)[sample_idx]
    return dict(zip(le.inverse_transform(np.arange(len(proba))), np.round(proba, 4)))

def predict_torch(X):
    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.model(x)

    classes = np.load(os.path.join(MODEL_DIR, "label_encoder_classes_pt.npy"), allow_pickle=True)
    le = LabelEncoder()
    le.classes_ = classes
    scaler = load(os.path.join(MODEL_DIR, "scaler_pt.pkl"))
    X_scaled = scaler.transform(X)
    model = MLP(X.shape[1], len(classes))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "torch_ssn_model.pt"), map_location='cpu'))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(X_scaled[sample_idx], dtype=torch.float32).unsqueeze(0)
        outputs = model(input_tensor)
        proba = F.softmax(outputs, dim=1).numpy()[0]
    return dict(zip(le.inverse_transform(np.arange(len(proba))), np.round(proba, 4)))

# === Uruchomienie predykcji ===
with st.spinner("🔍 Obliczanie predykcji..."):
    if model_choice == "MLP":
        results = predict_mlp(X_test)
    elif model_choice == "Random Forest":
        results = predict_rf(X_test)
    elif model_choice == "TensorFlow":
        results = predict_tf(X_test)
    elif model_choice == "PyTorch":
        results = predict_torch(X_test)

# === Wizualizacja wyników ===
st.subheader("📈 Prawdopodobieństwa klas")
st.bar_chart(
    pd.DataFrame.from_dict(results, orient="index", columns=["Prawdopodobieństwo"])
    .sort_values(by="Prawdopodobieństwo", ascending=False)
)
