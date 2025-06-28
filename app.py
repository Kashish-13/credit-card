import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

st.title("🕵️‍♂️ Credit Card Fraud Detection using DBSCAN")

# Upload CSV file
uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv", "zip"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Dataset")
        st.write(df.head())

        # Drop unneeded columns
        if "Time" in df.columns:
            df.drop("Time", axis=1, inplace=True)

        # Save 'Class' if available for validation later
        y_true = df["Class"] if "Class" in df.columns else None
        if "Class" in df.columns:
            df.drop("Class", axis=1, inplace=True)

        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        # Dimensionality Reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        # DBSCAN clustering
        eps = st.slider("Select DBSCAN eps value", 0.1, 5.0, step=0.1, value=0.5)
        min_samples = st.slider("Select min_samples", 1, 20, value=5)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(reduced_data)

        # Show cluster summary
        st.subheader("📌 Clustering Results")
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)
        st.write(f"Clusters Found: **{n_clusters}**")
        st.write(f"Potential Fraudulent Transactions (Noise): **{n_outliers}**")

        # Visualization
        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='rainbow', s=10)
        ax.set_title("DBSCAN Clustering (PCA-reduced)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.pyplot(fig)

        # Add clustering output to data
        output_df = pd.DataFrame(df)
        output_df["Cluster"] = labels
        output_df["Is_Fraud"] = output_df["Cluster"] == -1

        # Show detected frauds
        st.subheader("⚠️ Detected Anomalies (Potential Frauds)")
        st.write(output_df[output_df["Is_Fraud"] == True])

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a `creditcard.csv` file to begin.")
