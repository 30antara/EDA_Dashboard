import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Advanced EDA Dashboard", layout="wide")
st.title("Advanced Interactive EDA Dashboard")

st.write("Upload any CSV file to explore, visualize, clean, and download your data.")


# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("Dataset Info")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Column Types:")
        st.write(df.dtypes)

    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())
    
    # Summary Stats
    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include='all'))

    # Numeric & Categorical Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.markdown("---")
    st.subheader("Distributions & Visualizations")

    # Histogram
    if numeric_cols:
        st.write("> Histogram")
        col = st.selectbox("Select numeric column:", numeric_cols, key="hist")
        fig, ax = plt.subplots(figsize=(8,5), dpi=150)
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig, use_container_width=False)

    # Categorical Countplot
    if categorical_cols:
        st.write("> Countplot (Categorical)")
        col = st.selectbox("Select categorical column:", categorical_cols, key="countplot")
        fig, ax = plt.subplots(figsize=(8,5), dpi=150)
        sns.countplot(x=df[col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=False)

    # Scatter Plot
    if len(numeric_cols) >= 2:
        st.write("> Scatter Plot")
        colx = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
        coly = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")
        fig, ax = plt.subplots(figsize=(8,5), dpi=150)
        sns.scatterplot(x=df[colx], y=df[coly], ax=ax)
        st.pyplot(fig, use_container_width=False)

    # Box Plot (for outliers)
    if numeric_cols:
        st.write("> Box Plot")
        col = st.selectbox("Select numeric column:", numeric_cols, key="box")
        fig, ax = plt.subplots(figsize=(8,5), dpi=150)
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig, use_container_width=False)

    # Pairplot
    if st.checkbox("Show Pairplot (slow for large datasets)"):
        fig = sns.pairplot(df[numeric_cols])
        st.pyplot(fig, use_container_width=False, dpi=150)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,5), dpi=150)
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig, use_container_width=False)

    # Outlier Detection
    st.markdown("---")
    st.subheader("Outlier Detection (IQR Method)")
    
    selected_col = st.selectbox("Choose column for outlier detection:", numeric_cols)

    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[selected_col] < lower) | (df[selected_col] > upper)]

    st.write(f"Found {outliers.shape[0]} outliers.")
    st.dataframe(outliers.head())

    # Data Cleaning
    st.markdown("---")
    st.subheader("Data Cleaning")

    if st.button("Remove rows with missing values"):
        df = df.dropna()
        st.success("Missing values removed.")

    if st.button("Fill missing numeric values with mean"):
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success("Numeric NaNs filled with mean.")

    if st.button("Remove Outliers (selected column)"):
        df = df[(df[selected_col] >= lower) & (df[selected_col] <= upper)]
        st.success("Outliers removed.")

    # Data Download
    st.markdown("---")
    st.subheader("Download Cleaned Dataset")

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="Download CSV",
        data=buffer,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin.")
