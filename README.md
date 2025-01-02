# Gram Gold/TL Time Series Analysis

This repository contains the implementation and analysis for the project **"Gram Gold/TL Time Series Analysis"** conducted by **Egehan Sözen** and **Emirhan Söbüğlu** as part of a study in Bilişim Sistemleri Mühendisliği. The project explores time series forecasting of gold prices using a variety of machine learning and statistical models.

## Authors

- **Egehan Sözen**
  - Email: [egehansozenn@gmail.com](mailto:egehansozenn@gmail.com)
  - ID: 221307013
- **Emirhan Söbüğlu**
  - Email: [emirsbgl@gmail.com](mailto:emirsbgl@gmail.com)
  - ID: 221307014

## Abstract

This study focuses on time series analysis for gold price forecasting using five models:

- Linear Regression
- XGBoost
- Temporal Fusion Transformer (TFT)
- Informer
- ARIMA

The dataset, containing 3488 records scraped using Selenium, was preprocessed and analyzed in Python on Google Colab. Key evaluation metrics include MSE, MAPE, MAE, RMSE, R-Squared, and training/inference times. The study provides insights into the strengths and limitations of different models for time series forecasting.

## Keywords

- Gold price forecasting
- Transformer models
- Time series analysis
- Machine learning
- Python
- Web scraping

---

## Project Overview

### 1. Introduction

Gold prices play a crucial role in economies and financial markets. This project aims to:

- Predict Gram Altın/TL prices using both traditional and advanced machine learning models.
- Evaluate model performance based on accuracy and computation time.

### 2. Methodology

#### A. Data Collection and Preprocessing

- **Data Source:** [Investing.com](https://www.investing.com)
- **Scraping Tool:** Selenium
- **Technical Indicators:** Moving Averages, Relative Strength Index (RSI), and others
- **Normalization:** Standardized data for model training
- **Storage:** Data saved in CSV format

#### B. Models Used

1. **Linear Regression**: Establishes a linear relationship between features and the target variable.
2. **XGBoost**: Ensemble learning model known for its robustness and speed.
3. **Temporal Fusion Transformer (TFT)**: Designed for time series forecasting with temporal dependencies.
4. **Informer**: Optimized for long-term dependency modeling in time series.
5. **ARIMA**: Classical statistical model for univariate time series analysis.

---

## Results and Discussion

### A. Evaluation Metrics

- **MSE (Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R-Squared (R²)**
- **Time Efficiency**: Training and inference times

### B. Model Performance

| Model              | MSE       | MAE    | RMSE   | MAPE    | R²     | Training Time | Inference Time |
|--------------------|-----------|--------|--------|---------|---------|---------------|----------------|
| Linear Regression  | 424.34    | 9.83   | 20.60  | 2.28%   | 1.00    | 0.3 s         | 0.00026 s      |
| XGBoost            | 133.34    | 5.58   | 11.55  | 1.23%   | 1.00    | 1.34 s        | 0.00516 s      |
| TFT                | 579568.06 | 426.02 | 761.29 | 76.37%  | -0.23   | 117.13 s      | 0.08350 s      |
| Informer           | 94.52     | 7.80   | 9.72   | 2.98%   | -1.38   | 40.74 s       | 0.51 s         |
| ARIMA              | 0.12      | 0.26   | 0.34   | -1.38%  | -1.38   | 6.06 s        | 0.01470 s      |

### C. Key Findings

1. **Accuracy:** Informer achieved the lowest MSE and RMSE.
2. **Efficiency:** Linear Regression and XGBoost were fastest in training and inference times.
3. **Interpretability:** ARIMA offered simple and interpretable results but struggled with multivariate data.

---

## Conclusion

The study demonstrates that advanced transformer models like TFT and Informer provide high accuracy in gold price forecasting. However, traditional models like Linear Regression and ARIMA remain valuable for simplicity and speed. Future work could explore hybrid models and the inclusion of external economic indicators.

---

## Acknowledgments

- **Data Source:** [Investing.com](https://www.investing.com)
- **Development Tools:** Google Colab, Python libraries (Selenium, Pandas, etc.)

---

## References

1. [DataCamp - Time Series Analysis Tutorial](https://www.datacamp.com/tutorial/time-series-analysis-tutorial)
2. [PyTorch Documentation](https://pytorch.org)
3. [Pandas Documentation - 10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
4. [Veri Bilimi Okulu - Time Series Analysis with Python](https://www.veribilimiokulu.com/python-ile-zaman-serisi-analizi/)
5. [Informer GitHub Repository](https://github.com/zhouhaoyi/Informer2020)

---
