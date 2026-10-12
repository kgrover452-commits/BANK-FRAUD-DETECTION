💳 Bank Fraud Detection Using Machine Learning
A machine learning based fraud detection system that identifies potentially fraudulent financial transactions using transaction patterns, feature engineering, and classification models.
The project also includes a Streamlit web application for real-time transaction risk prediction.
📌 Problem Statement
Financial fraud can cause significant losses to banks and customers. Since fraudulent transactions are much less frequent than legitimate transactions, fraud detection is a highly imbalanced classification problem.
The objective of this project is to build a machine learning system that can identify potentially fraudulent transactions while maintaining a high fraud detection rate.
🎯 Objectives
- Analyze financial transaction data
- Perform Exploratory Data Analysis (EDA)
- Handle severe class imbalance
- Create meaningful features from transaction data
- Train and compare multiple machine learning models
- Tune model hyperparameters
- Tune the classification threshold
- Save the trained model using Joblib
- Build a Streamlit application for fraud prediction
📊 Dataset
The dataset contains approximately 6.36 million transactions and 11 original features.
Important features include:
- step – Transaction time step
- type – Type of transaction
- amount – Transaction amount
- nameOrig – Sender ID
- oldbalanceOrg – Sender balance before transaction
- newbalanceOrig – Sender balance after transaction
- nameDest – Receiver ID
- oldbalanceDest – Receiver balance before transaction
- newbalanceDest – Receiver balance after transaction
- isFraud – Fraud indicator
- isFlaggedFraud – Existing fraud flag
The dataset contains approximately 8,213 fraudulent transactions, making it a highly imbalanced classification problem.
The original dataset is not included in this repository because of its large size.
🔎 Exploratory Data Analysis
The following analysis was performed:
- Fraud rate by transaction type
- Transaction amount distribution
- Log-transformed transaction amount
- Identification of high-value transactions
- Transaction hour analysis
- Night-time transaction analysis
- Correlation analysis
- Analysis of numerical transaction features
The analysis showed that fraudulent transactions represent only a very small percentage of the overall transactions, making class imbalance an important consideration during model development.
⚙️ Feature Engineering
Feature engineering was performed to extract useful information from the raw transaction data.
1. Log Transaction Amount
df['log_amount'] = np.log1p(df['amount'])
Transaction amounts are highly skewed, with some transactions having extremely large values.
The log transformation reduces the effect of extreme values and provides a more stable representation of transaction amount.
2. Sender Balance Difference
df['balance_diff_orig'] = (
    df['oldbalanceOrg'] - df['newbalanceOrig']
)
This feature represents the change in the sender's balance during the transaction.
It helps capture how much money was removed from the sender's account.
3. Receiver Balance Difference
df['balance_diff_dest'] = (
    df['newbalanceDest'] - df['oldbalanceDest']
)
This feature represents the change in the receiver's balance.
It provides additional information about the movement of money during the transaction.
4. High-Value Transaction
p99 = df['amount'].quantile(0.99)

df['is_high_amount'] = (
    df['amount'] > p99
).astype(int)
The 99th percentile was used to identify unusually large transactions.
Transactions above this threshold were marked as high-value transactions.
5. Transaction Hour
df['hour'] = df['step'] % 24
The transaction step was converted into an approximate hour of the day.
This allows the model to capture possible time-based transaction patterns.
6. Night Transaction
night_hours = [0, 1, 2, 3, 4, 5, 22, 23]

df['is_night'] = (
    df['hour'].isin(night_hours)
).astype(int)
Transactions occurring during selected night hours were marked using a binary feature.
7. Transaction Type Encoding
le = LabelEncoder()

df['type_enc'] = le.fit_transform(
    df['type']
)
The categorical transaction type was converted into numerical values so that it could be used by machine learning models.
8. Removal of Original Balance Features
After creating the balance-difference features, the original balance columns were removed to use the engineered representations.
🧹 Data Preprocessing
The following preprocessing steps were performed:
- Missing value handling
- Label encoding of transaction type
- Removal of identifier columns
- Feature engineering
- Stratified train-test split
- Robust scaling of numerical features
Stratification was used to preserve the fraud-to-legitimate transaction ratio in both training and testing data.
🤖 Machine Learning Models
Three machine learning classification models were trained and evaluated:
1. Logistic Regression
2. Random Forest
3. XGBoost
Initial Model Performance
Model	ROC-AUC	PR-AUC
Logistic Regression	0.9775	0.5506
Random Forest	0.9976	0.8414
XGBoost	0.9980	0.8757


XGBoost achieved the highest initial PR-AUC among the evaluated models.
🔧 Hyperparameter Tuning
The models were further tuned using parameters such as:
- Number of estimators
- Maximum tree depth
- Learning rate
- Minimum child weight
- Subsampling
- Column sampling
- Class weighting
For XGBoost, scale_pos_weight was used to account for the severe class imbalance.
Tuned Model Performance
Model	ROC-AUC	PR-AUC
Logistic Regression	0.9775	0.5504
Random Forest	0.9978	0.8329
XGBoost	0.9982	0.8617


🎚️ Threshold Tuning
Because fraud detection is an imbalanced classification problem, the default probability threshold of 0.5 was not used directly.
Precision-Recall analysis was used to select a threshold that targeted approximately 90% recall while maximizing precision among the evaluated thresholds.
The selected threshold was approximately:
0.908
This allows the application to control the trade-off between detecting fraudulent transactions and generating false alerts.
💾 Model Saving
The trained model and required preprocessing objects were saved using Joblib.
The saved artifact contains:
- Trained model
- Decision threshold
- Feature names
- RobustScaler
- LabelEncoder
- High-amount threshold
This allows the Streamlit application to load the trained pipeline without retraining the model.
🌐 Streamlit Application
A Streamlit web application was developed to provide an interactive interface for fraud prediction.
The application allows users to enter:
- Transaction type
- Transaction amount
- Sender balance before transaction
- Sender balance after transaction
- Receiver balance before transaction
- Receiver balance after transaction
- Transaction step
The application then performs the same feature engineering and preprocessing used during model training and returns:
- Fraud probability
- Fraud / legitimate classification
Application Flow
User Input
     ↓
Feature Engineering
     ↓
Data Preprocessing
     ↓
Trained ML Model
     ↓
Fraud Probability
     ↓
Decision Threshold
     ↓
Fraud / Legitimate Result
📁 Project Structure
BANK-FRAUD-DETECTION/
│
├── Bankfraud.ipynb
├── app.py
├── best_fraud_model_tuned.pkl
├── README.md
└── .gitignore
The original dataset is excluded from the repository because of its large size.
🚀 How to Run
1. Clone the repository
git clone https://github.com/kgrover452-commits/BANK-FRAUD-DETECTION.git
2. Navigate to the project
cd BANK-FRAUD-DETECTION
3. Install dependencies
pip install pandas numpy scikit-learn xgboost joblib streamlit
4. Run the Streamlit application
streamlit run app.py
The application will open in your browser.
🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook
🔮 Future Improvements
- Improve precision at high recall
- Experiment with anomaly detection techniques
- Perform systematic hyperparameter optimization
- Add SHAP-based model explainability
- Improve the Streamlit dashboard
- Add transaction risk levels
- Deploy the application online
📌 Project Status
Completed
- Data analysis
- Feature engineering
- Data preprocessing
- Multiple machine learning models
- Model comparison
- Hyperparameter tuning
- Threshold tuning
- Model serialization
- Streamlit application
Future Work
- Advanced model explainability
- Further precision-recall optimization
- Online deployment