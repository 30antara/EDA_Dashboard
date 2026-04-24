# 📊 EDA Dashboard

An interactive, end-to-end Exploratory Data Analysis and ML tool built with Streamlit. Upload any CSV, clean it, visualize it, and train a baseline model — all in one place.

---

## 🚀 Features

**Upload & Profile**
- Auto-detects column types, missing values, duplicates
- Full column summary (dtype, null count, unique values)

**Data Cleaning**
- Handle missing values: drop, mean, median, or mode fill
- Remove duplicate rows
- Download cleaned dataset as CSV

**Visualizations**
- Histogram with adjustable bin count
- Boxplot with optional category grouping
- Scatter plot with color encoding
- Correlation heatmap (diverging colorscale)
- Bar chart for categorical columns (top-N view)

**ML Model (Logistic Regression)**
- Adjustable train/test split
- Metrics: Accuracy, Precision, Recall, F1
- Confusion matrix
- Full classification report
- Feature importance chart (top 20 features)

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| ML | Scikit-learn |

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/30antara/EDA_Dashboard.git
cd EDA_Dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run eda_dashboard.py
```

Opens at `http://localhost:8501`

---

## 📁 Project Structure

```
EDA_Dashboard/
├── eda_dashboard.py        # Main application
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Theme configuration
└── README.md
```

---

## 🔗 Links

- **GitHub:** [github.com/30antara](https://github.com/30antara)
- **LinkedIn:** [linkedin.com/in/antarasinghal](https://linkedin.com/in/antarasinghal)
