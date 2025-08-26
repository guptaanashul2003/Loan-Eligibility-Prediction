# 💳 Loan Eligibility Prediction

This project predicts whether a loan application will be **approved or rejected** using machine learning. It applies classification techniques on a loan dataset with features like income, loan amount, marital status, gender, and credit history.  

---

## 📌 Project Overview
- Dataset: Loan Eligibility dataset (commonly used in ML practice).  
- Tools: **Python, Pandas, Seaborn, Matplotlib, Scikit-learn**.  
- Goal: Build a classification model to predict **Loan Status** (Approved/Not Approved).  

---

## 🛠️ Steps Performed
1. **Data Cleaning**
   - Handled missing values.  
   - Encoded categorical variables (Gender, Married, Loan Status).  

2. **Preprocessing**
   - Train-test split (80/20).  
   - Standardized numeric features.  

3. **Model Training**
   - Applied **Logistic Regression** as baseline model.  

4. **Evaluation**
   - Accuracy score.  
   - Confusion Matrix visualization.  
   - Classification Report (Precision, Recall, F1-score).  

---

## 🔑 Key Insights
- **Credit History** plays the biggest role in loan approval.  
- Applicant Income and Loan Amount also influence eligibility.  
- Logistic Regression achieved **decent accuracy**; advanced models can improve predictions.  

---

## 📂 Files
- `Loan_Eligibility.ipynb` → Full Jupyter/Colab notebook with code & results  
- `README.md` → Documentation  

---

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/Loan-Eligibility-Prediction.git
pip install pandas seaborn matplotlib scikit-learn
