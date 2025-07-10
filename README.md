# Personalized Healthcare Recommendations: Machine Learning Project

## 🌟 Project Overview

This project aims to develop a machine learning model that provides personalized healthcare recommendations based on individual patient data, leveraging a dataset common in Recency, Frequency, Monetary, and Time (RFMT) analysis, which can reflect aspects of patient engagement or interaction patterns within a healthcare system. The primary goal is to predict a `Class` label that translates into a actionable healthcare recommendation (e.g., 'No immediate action needed' vs. 'Consult with a healthcare professional').

## ✨ Features & Technologies

*   **Domain:** Healthcare, Data Analytics, Machine Learning
*   **Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, LightGBM, Streamlit
*   **Tools:** Jupyter Notebook (for development), VS Code (for development), Git/GitHub (for version control and hosting)
*   **Difficulty:** Advanced

## 📂 Dataset

The dataset used in this project comprises **748 entries** and **5 columns**:
- `Recency`: *(Describe what Recency represents in your context, e.g., time since last patient interaction in days/months)*
- `Frequency`: *(Describe what Frequency represents, e.g., total number of patient interactions)*
- `Monetary`: *(Describe what Monetary represents, e.g., total value of patient's contributions or spending in the system)*
- `Time`: *(Describe what Time represents, e.g., total duration patient has been engaged, or cumulative time units)*
- `Class`: The target variable, representing different recommendation types or health outcomes. In this project, `Class 0` (No action needed) and `Class 1` (Consult a professional).

*(If the dataset is publicly available, provide a direct link to it or a source. If it was from an internal source, you can state that. e.g.,)*
**Source:** [Click here to download data set - _Replace with actual download link or dataset name/origin_]

## 🚀 Project Steps & Highlights

### 1. Data Exploration & Visualization (EDA)

Performed an in-depth EDA to understand the distribution, statistics, and potential correlations within the RFMT features.
- Initial data scan confirmed all features as numerical with no missing values (after initial preprocessing).
- Explored distributions of `Recency`, `Frequency`, `Monetary`, and `Time`.
- Analyzed the class distribution for the `Class` target variable, identifying a **class imbalance** (`Class 0` was 570 entries, `Class 1` was 178 entries).

### 2. Data Preprocessing

- **Feature and Target Separation:** Identified `Recency`, `Frequency`, `Monetary`, `Time` as features (X) and `Class` as the target (y).
- **Standard Scaling:** Applied `StandardScaler` to all numerical features. This normalizes the feature values to a standard range, crucial for many machine learning algorithms.
- **Stratified Train-Test Split:** Split the dataset into 80% training and 20% testing sets using `stratify=y` to maintain the original class distribution in both sets, which is vital for imbalanced datasets.

### 3. Model Selection & Training

The project employed various machine learning classification models and strategically addressed class imbalance for improved minority class prediction.

- **Primary Model (`RandomForestClassifier`):** A Random Forest Classifier was chosen as the main model for the predictive pipeline. It was configured with `class_weight='balanced'` to explicitly handle the class imbalance.
- **Advanced Models for Cross-Validation:**
    - `XGBoost (XGBClassifier)`
    - `LightGBM (LGBMClassifier)`
    These models were cross-validated on the full dataset with their respective class-weighting strategies (`scale_pos_weight` for XGBoost, `class_weight='balanced'` for LightGBM).

#### Cross-Validation Performance:

| Model                         | Average Accuracy | Recall (Class 1) |
| :---------------------------- | :--------------- | :--------------- |
| RandomForest (Balanced)       | **~0.61**        | **~0.31**        |
| XGBoost (Scaled Pos Weight)   | ~0.56            | ~0.34            |
| LightGBM (Balanced)           | **~0.57**        | **~0.46**        |

*Interpretation:* While RandomForest showed a slightly higher overall accuracy in CV, **LightGBM significantly improved Recall for `Class 1` (the minority class) to ~0.46**, demonstrating its effectiveness in identifying potentially critical cases despite the imbalance. This makes LightGBM a strong candidate for a personalized recommendation system focused on risk identification.

### 4. Model Evaluation

The trained `RandomForestClassifier` pipeline was evaluated on the held-out test set.

- **Overall Accuracy:** `~0.71`
- **Confusion Matrix:**
    ```
    [[90 24]
     [19 17]]
    ```
- **Classification Report Highlights:**
    - `Class 0` (Majority): High precision (0.83), high recall (0.79).
    - `Class 1` (Minority): **Improved precision (0.41)** and **recall (0.47)**, reflecting the effect of class weighting.
- **ROC-AUC:** `~0.71`

### 5. Personalized Recommendation System Implementation

A Python function `generate_recommendations` was implemented to provide personalized advice.
- It takes raw patient RFMT data as input.
- Uses the trained model pipeline to predict the `Class`.
- Maps the predicted `Class` to a human-readable recommendation:
    - `Class 0`: 'No immediate action needed; maintain current health routine.'
    - `Class 1`: 'Recommendation: Consult with a healthcare professional for further evaluation/check-up.'

## ☁️ Deployment (Streamlit Web Dashboard)

To make this model interactive and demonstrate its potential for practical use, a simple web dashboard was built using **Streamlit**.

- **Purpose:** Allows users (e.g., healthcare professionals) to input patient RFMT data via sliders and instantly receive a predicted healthcare recommendation.
- **Access:**
    1.  Ensure you have `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `xgboost`, `lightgbm` installed (`pip install streamlit pandas numpy scikit-learn joblib xgboost lightgbm`).
    2.  Run the main project script (your `.ipynb` or `.py` file) once to train and save the `recommendation_pipeline.joblib` model.
    3.  Navigate to the directory containing `app.py` and the saved `.joblib` file in your terminal.
    4.  Run: `streamlit run app.py`

**Screenshot of the Streamlit App:**
*(Add a screenshot of your deployed Streamlit app here for a powerful visual demonstration!)*
![Streamlit Dashboard Screenshot](path/to/your/streamlit_screenshot.png)

## 🎯 Conclusion & Implications

This project successfully developed a personalized healthcare recommendation system using a common RFMT dataset.
- The comprehensive EDA provided critical insights into the data's characteristics and class imbalance.
- Robust preprocessing techniques prepared the data effectively for machine learning.
- By leveraging class weighting strategies, the models (particularly LightGBM and RandomForest with balanced weights) demonstrated an improved ability to identify patients belonging to the minority class (`Class 1`), which is crucial for proactive healthcare recommendations.
- The developed system, prototype via a Streamlit dashboard, showcases the potential of data-driven insights in offering personalized advice to patients.

**Potential Implications:** This type of system could help:
- Prioritize patient outreach for check-ups based on behavioral patterns.
- Tailor lifestyle or intervention recommendations.
- Optimize resource allocation by identifying patients most likely to benefit from certain recommendations.

## ⚙️ Future Work

-   **Hyperparameter Tuning:** Fine-tune model parameters for `RandomForestClassifier`, `XGBoost`, and `LightGBM` using `GridSearchCV` or `Optuna` to potentially further boost performance.
-   **Explore Advanced Imbalance Techniques:** Implement more sophisticated methods like ADASYN or investigate different sampling strategies (e.g., combining oversampling and undersampling).
-   **Model Interpretability (XAI):** Integrate libraries like SHAP or LIME to provide clear explanations for *why* a specific recommendation is generated for an individual patient, which is vital in a healthcare context.
-   **External Validation:** Test the model on new, independent datasets to ensure generalizability.
-   **A/B Testing Framework:** Conceptually outline how this system could be A/B tested in a live environment to measure its real-world impact.
-   **Database Integration:** Connect the system to a real database for dynamic patient data.

## 🤝 Contribution

This project was developed by [Your Name].

---
