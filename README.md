# Advanced Interactive EDA Dashboard

An interactive **Exploratory Data Analysis (EDA) dashboard** built using **Python** and **Streamlit** to explore, visualize, clean, and download structured datasets through a user-friendly web interface.


## Features
- Upload and preview CSV datasets
- Automated dataset profiling:
  - Shape, data types, missing values
  - Descriptive statistics
- Interactive visualizations:
  - Histograms
  - Box plots
  - Scatter plots
  - Correlation heatmaps
  - Pair plots
- Outlier detection using the IQR method
- Interactive data cleaning:
  - Missing value handling
  - Outlier removal
- Download cleaned datasets as CSV


## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn  
- **Framework:** Streamlit  

---

## To run locally

1. Clone the repository:
   git clone https://github.com/30antara/EDA_Dashboard.git
   
2. Navigate to project director
   cd EDA_Dashboard
   
3. Install dependencies
   pip install -r requirements.txt
   
4. Run the Streamlit app
   streamlit run app.py


## Project Motivation
This project was developed to support data quality assessment, exploratory analysis, and preprocessing as part of machine learning and advanced data analysis workflows.

## Future Enhancements
  - Performance optimization for large datasets
  - Support for text-based datasets and NLP-oriented EDA
  - Enhanced reporting and evaluation features



