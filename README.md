# Diabetes Screening Classifier: Optimizing Recall on Imbalanced Data

## 📌 Project Overview
This project builds and evaluates several machine learning classification models to predict the onset of diabetes based on diagnostic measurements. Because this is a medical screening task, the primary objective is to **maximize recall for the positive (diabetic) class**. In a clinical setting, minimizing false negatives (failing to identify a diabetic patient) is far more critical than overall accuracy.

## 📊 Dataset
The analysis uses the **Pima Indians Diabetes Dataset**, which contains 768 patient records and 8 diagnostic features:
* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`

**Class Imbalance:** The dataset is moderately imbalanced, with 500 negative (non-diabetic) and 268 positive (diabetic) cases.

## 🛠️ Data Preprocessing & Engineering
To ensure robust model training and prevent data leakage, the following preprocessing steps were implemented:
1. **Handling Missing Values:** Physiologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were treated as missing data. 
2. **Median Imputation:** Missing values were imputed using the median of the *training set* only.
3. **Feature Engineering:** A binary `Insulin_missing` flag was created to preserve the predictive signal of the originally missing insulin data (which accounted for ~48% of the dataset).
4. **Scaling:** Features were standardized using `StandardScaler` to optimize distance-based and linear algorithms.

## 🤖 Modeling & Evaluation
Several classification algorithms were trained and evaluated. To address the class imbalance, `class_weight='balanced'` was applied to supported models.

**Models Tested:**
* Logistic Regression
* Support Vector Classifier (SVC)
* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* AdaBoost
* Gaussian Naive Bayes

### Cross-Validation Strategy
While tree-based ensembles (Random Forest, AdaBoost) appeared highly competitive on a single train/test split, **5-fold Cross-Validation** (scored specifically on `recall`) was utilized to obtain a more reliable estimate of model stability. 

**Cross-Validation Results (Recall):**
* **SVC: 0.7612 (+/- 0.0546) 🏆**
* Logistic Regression: 0.7434 (+/- 0.0747)
* AdaBoost: 0.6082 (+/- 0.0955)
* Random Forest: 0.6078 (+/- 0.0905)
* KNN: 0.5902 (+/- 0.0571)
* Decision Tree: 0.5765 (+/- 0.1260)

## 🏁 Conclusion
Cross-validation demonstrated that the **Support Vector Classifier (SVC)** achieved the best and most stable recall (~0.76) for the diabetic class. It significantly outperformed the tree-based ensembles, which suffered from instability across different folds. Hyperparameter tuning via `GridSearchCV` confirmed that the default SVC configuration with balanced class weights was already near-optimal. 

**Final Model Performance (Held-out Test Set):**
* **Algorithm:** SVC (`class_weight='balanced'`)
* **Recall (Diabetic Class):** ~76%
* **Overall Accuracy:** ~73%

## 💻 Technologies Used
* **Python**
* **pandas & NumPy:** Data manipulation and mathematical operations
* **scikit-learn:** Model building, preprocessing, and evaluation
* **Matplotlib & Seaborn:** Data visualization
