import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# PAGE CONFIG
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

# HEADER
st.markdown("""
# 📊 **EDA Dashboard**
### Upload → Clean → Visualize → Model  
---
""")

# SIDEBAR NAVIGATION
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Upload Data", "Cleaning", "Visualizations", "ML Model"]
)

# GLOBAL CONTAINER
if "df" not in st.session_state:
    st.session_state.df = None


# ─────────────────────────────────────────
# UPLOAD PAGE
# ─────────────────────────────────────────
if page == "Upload Data":

    st.subheader("📁 Upload Your CSV File")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.success("✅ Data uploaded successfully!")

        st.write("### 📌 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
        with c3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))
        with c4:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.write("### 📋 Column Summary")
        summary = pd.DataFrame({
            "dtype": df.dtypes,
            "non_null": df.notnull().sum(),
            "null": df.isnull().sum(),
            "null_%": (df.isnull().sum() / len(df) * 100).round(2),
            "unique": df.nunique()
        })
        st.dataframe(summary, use_container_width=True)


# ─────────────────────────────────────────
# CLEANING PAGE
# ─────────────────────────────────────────
elif page == "Cleaning":

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
    else:
        df = st.session_state.df.copy()
        st.subheader("🧹 Data Cleaning Tools")

        # Missing values
        st.write("### Missing Value Summary")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found!")
        else:
            st.dataframe(missing.rename("Missing Count"), use_container_width=True)

        st.write("### Handle Missing Values")
        method = st.radio("Choose a cleaning method:", [
            "None", "Drop missing rows", "Fill with mean",
            "Fill with median", "Fill with mode"
        ])

        if method == "Drop missing rows":
            before = len(df)
            df = df.dropna()
            st.success(f"Dropped {before - len(df)} rows with missing values.")
        elif method == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("Missing values filled with column mean.")
        elif method == "Fill with median":
            df = df.fillna(df.median(numeric_only=True))
            st.success("Missing values filled with column median.")
        elif method == "Fill with mode":
            for col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values filled with column mode.")

        # Drop duplicates option
        st.write("### Handle Duplicates")
        n_dupes = df.duplicated().sum()
        st.write(f"Duplicate rows found: **{n_dupes}**")
        if n_dupes > 0 and st.button("Drop Duplicate Rows"):
            df = df.drop_duplicates()
            st.success(f"Dropped {n_dupes} duplicate rows.")

        st.session_state.df = df

        # Download cleaned dataset
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned Dataset", csv, "cleaned_data.csv", "text/csv")

        st.write("### Preview")
        st.dataframe(df.head(), use_container_width=True)


# ─────────────────────────────────────────
# VISUALIZATION PAGE
# ─────────────────────────────────────────
elif page == "Visualizations":

    if st.session_state.df is None:
        st.warning("⚠️ Upload a dataset first.")
    else:
        df = st.session_state.df
        st.subheader("📊 Visualizations")

        chart_type = st.selectbox(
            "Choose Plot Type",
            ["Histogram", "Boxplot", "Scatter Plot", "Correlation Heatmap", "Bar Chart (Categorical)"]
        )

        if chart_type == "Histogram":
            col = st.selectbox("Select column:", df.select_dtypes(include=np.number).columns)
            bins = st.slider("Number of bins:", 5, 100, 30)
            fig = px.histogram(df, x=col, nbins=bins, opacity=0.85)
            fig.update_traces(marker_line_color="black", marker_line_width=1.2)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Boxplot":
            col = st.selectbox("Select column:", df.select_dtypes(include=np.number).columns)
            color_col = st.selectbox(
                "Group by (optional):",
                ["None"] + list(df.select_dtypes(include="object").columns)
            )
            fig = px.box(df, y=col, color=None if color_col == "None" else color_col)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            num_cols = df.select_dtypes(include=np.number).columns
            x = st.selectbox("X-axis:", num_cols)
            y = st.selectbox("Y-axis:", num_cols)
            color_col = st.selectbox(
                "Color by (optional):",
                ["None"] + list(df.select_dtypes(include="object").columns)
            )
            fig = px.scatter(df, x=x, y=y, color=None if color_col == "None" else color_col, opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                st.warning("Need at least 2 numeric columns for a correlation heatmap.")
            else:
                fig = px.imshow(
                    numeric_df.corr().round(2),
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart (Categorical)":
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) == 0:
                st.warning("No categorical columns found in the dataset.")
            else:
                col = st.selectbox("Select categorical column:", cat_cols)
                top_n = st.slider("Show top N categories:", 5, 30, 10)
                counts = df[col].value_counts().head(top_n).reset_index()
                counts.columns = [col, "count"]
                fig = px.bar(counts, x=col, y="count", text="count")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# ML MODEL PAGE
# ─────────────────────────────────────────
elif page == "ML Model":

    if st.session_state.df is None:
        st.warning("⚠️ Upload data first.")
    else:
        df = st.session_state.df
        st.subheader("🤖 Machine Learning Model (Logistic Regression)")

        target = st.selectbox("Select target column:", df.columns)
        test_size = st.slider("Test set size (%):", 10, 40, 20) / 100

        X = df.drop(columns=[target])
        y = df[target]

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        num_cols = X.select_dtypes(exclude="object").columns.tolist()

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ], remainder="passthrough")

        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("logreg", LogisticRegression(max_iter=1000))
        ])

        if st.button("Train Model"):
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # ── Metrics ──────────────────────────────
                st.write("### 📈 Model Performance")
                avg = "binary" if y.nunique() == 2 else "macro"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2f}")
                m2.metric("Precision", f"{precision_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                m3.metric("Recall",    f"{recall_score(y_test, y_pred, average=avg, zero_division=0):.2f}")
                m4.metric("F1 Score",  f"{f1_score(y_test, y_pred, average=avg, zero_division=0):.2f}")

                # ── Confusion Matrix ─────────────────────
                st.write("### 🔲 Confusion Matrix")
                labels = sorted(y.unique().tolist())
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm.tolist(),
                    x=[str(l) for l in labels],
                    y=[str(l) for l in labels],
                    colorscale="Blues",
                    showscale=True
                )
                fig_cm.update_layout(
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    yaxis_autorange="reversed"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # ── Classification Report ────────────────
                st.write("### 📋 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().round(2)
                st.dataframe(report_df, use_container_width=True)

                # ── Feature Importance ───────────────────
                st.write("### 🔍 Feature Importance (Logistic Regression Coefficients)")
                try:
                    logreg = model.named_steps["logreg"]
                    pre = model.named_steps["preprocess"]

                    # Get feature names after preprocessing
                    ohe_features = []
                    if cat_cols:
                        ohe_features = pre.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
                    feature_names = ohe_features + num_cols

                    if logreg.coef_.shape[0] == 1:
                        coefs = logreg.coef_[0]
                    else:
                        coefs = np.mean(np.abs(logreg.coef_), axis=0)

                    feat_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": np.abs(coefs)
                    }).sort_values("importance", ascending=False).head(20)

                    fig_imp = px.bar(
                        feat_df, x="importance", y="feature",
                        orientation="h",
                        labels={"importance": "Absolute Coefficient", "feature": "Feature"},
                        title="Top 20 Features by Importance"
                    )
                    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_imp, use_container_width=True)

                except Exception:
                    st.info("Feature importance could not be computed for this dataset.")

            except Exception as e:
                st.error(f"❌ Could not train model. Error: {e}")
