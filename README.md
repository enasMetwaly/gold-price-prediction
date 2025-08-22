# Gold Price Prediction using Machine Learning 🪙📈

This project predicts the future price of gold using historical data and machine learning.

## 🔍 Overview
- **Model**: Random Forest Regressor
- **Target**: GLD (Gold ETF price)
- **Features**: Silver (SLV), Oil (USO), Stock Market (SPX), EUR/USD
- **Goal**: Predict gold price based on financial indicators

## 📊 Dataset
- Source: Yahoo Finance or Kaggle
- Time Range: 2008–2023
- Columns: Date, SPX, GLD, USO, SLV, EUR/USD

## 🧪 How It Works
1. Data is loaded and cleaned
2. Correlation analysis shows Silver (SLV) is highly related to gold
3. Model is trained using Random Forest
4. Predictions are evaluated using MAE and R² score

## 🚀 How to Run
1. Open in [Google Colab](https://colab.research.google.com/)
2. Upload `gold_price_data.csv`
3. Run all cells
4. View prediction results

## 📈 Results
- Strong correlation between **Gold (GLD)** and **Silver (SLV)**: `0.87`
- Model accuracy: [To be filled]

## 🛠️ Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab

