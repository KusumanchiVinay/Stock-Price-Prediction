# 📈 Stock Price Prediction - Netflix (NFLX)

A comprehensive Machine Learning project to predict and analyze stock prices using the Netflix (NFLX) dataset. This project implements both **Regression** (to predict price values) and **Classification** (to predict price movement trends) using state-of-the-art algorithms like XGBoost and LightGBM.

## 📌 Project Overview
The goal of this project is to explore historical stock data, perform Exploratory Data Analysis (EDA), and build models that can forecast future prices and trends.

### Key Features:
- **Data Preprocessing**: Handling datetime objects, feature scaling, and data cleaning.
- **Exploratory Data Analysis (EDA)**: Visualizing price trends, volume, and correlation heatmaps.
- **Regression Models**: Predicting the exact 'Close' price.
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Support Vector Regressor (SVR)
  - LightGBM Regressor
- **Classification Models**: Predicting whether the price will go up or down.
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Classifier (SVC)

## 📂 Dataset
The project uses the `NFLX.csv` file, which contains historical stock data for Netflix.
- **Columns**: Date, Open, High, Low, Close, Adj Close, Volume.

## 🛠️ Installation & Setup

To run this project on your local system, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/KusumanchiVinay/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage
1. Ensure you have the `NFLX.csv` file in the project root directory.
2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```
3. Open `Stock Price Prediction.ipynb` and run the cells sequentially to see the analysis and model performance.

## 📊 Performance Metrics
- **Regression**: Evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
- **Classification**: Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC Curve.

## 📁 File Structure
```text
Stock-Price-Prediction/
│
├── Stock Price Prediction.ipynb   # Main analysis and modeling notebook
├── NFLX.csv                       # Historical stock data
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```

## 📝 Conclusion
This project demonstrates a complete pipeline for stock market analysis. The use of multiple models allows for a comparative study of which algorithm performs best for volatile financial data.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
