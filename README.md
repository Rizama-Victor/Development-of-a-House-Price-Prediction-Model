# Development of a Houseprice 🏡 Prediction 📈 Model

This repository contains the implementation of the project titled _"Development of a Houseprice Prediction Model"_, at the National Centre for Artificial Intelligence and Robotics (NCAIR), Nigeria.

---

## 🧠 Project Overview
The focus of this project was to develop a custom linear regression model from scratch using Python and train it on a house-price dataset. The goal was to create a predictive model capable of estimating future house prices based on learned patterns in the data.

## 🧰 Tools and Technologies Used

| **Tool / Library**                              | **Purpose in This Project**                                                                    |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **pandas**                                      | Used to load the dataset, clean the data, and handle tabular data operations.                           |
| **numpy**                                       | Performed numerical computations, array handling, and mathematical transformations during preprocessing. |
| **matplotlib.pyplot**                           | Helped to create basic visualizations to observe data patterns and feature relationships.                 |
| **seaborn**                                     | Used for statistical visualizations such as correlation heatmaps and distribution plots.                |
| **scipy.stats (zscore, boxcox)**                | Was used for outlier detection (Z-score) and data transformation (Box-Cox) to stabilize variance.           |
| **scikit-learn: train_test_split**              | Enabled splitting of the dataset into training and testing sets.                                               |
| **scikit-learn: StandardScaler**                | Helped scale numerical features so that the regression model trained more effectively.                 |
| **scikit-learn: OrdinalEncoder**                | Helped convert categorical variables into numerical form for model training.                           |
| **scikit-learn: LabelEncoder**                  | Encoded target or categorical features where necessary.                                          |
| **scikit-learn: Ridge & Lasso Regression**      | Was used to train regularized linear regression models for predicting house prices.                         |
| **scikit-learn: GridSearchCV**                  | Performed hyperparameter tuning and find the best model configuration.                            |
| **scikit-learn: mean_squared_error & r2_score** | Were used to evaluate model performance using MSE and R².                                                   |
| **joblib**                                      | Helped in saving the final trained model for future predictions.                                            |

## 🔁 Step by Step Procedure

1. **Loaded the data**  
   The house-prices dataset was loaded into a `pandas` DataFrame from a CSV file.

2. **Exploratory Data Analysis (EDA)**  
     Exploratory data checks were performed (head, info, describe) while missing values and obvious inconsistencies were identified and handled (dropped or imputed as appropriate).

3. **Removed outliers**  
   Outliers were detected and filtered using box plots and Z-score methods from `scipy.stats` to reduce the influence of anomalous observations.

4. **Encoded categorical variables**  
   Categorical columns were converted to numeric form using  `LabelEncoder` so they could be used by the linear model.

5. **Engineered features**  
   Useful features were created or derived (date parts, interaction terms, or domain-specific transformations) to make predictive patterns more explicit.

6. **Split the data**  
   The dataset was split into training and test subsets (typically with `train_test_split`) to preserve an unseen set for final evaluation. Next, feature matrices (`X_train`, `X_test`) and target vectors (`y_train`, `y_test`) were assembled from the processed DataFrame.

7. **Scaled numerical features**  
   Numeric features were standardized with `StandardScaler` to place them on comparable scales for model training.

8. **Trained the model**  
    The train set data was then fitted to the custom built Linear Regressor from scratct to enable the model understand the underlying patterns in the data.

9. **Generated predictions**  
    The trained model produced predictions on the test set for quantitative evaluation.

10. **Evaluated performance**  
    Regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root MSE (RMSE), and R² score were computed using `sklearn.metrics` to assess accuracy and fit.

11. **Visualized results**  
    Diagnostic plots specifically the predicted vs actual, was produced with `matplotlib` and `seaborn` to inspect model behavior and assumptions.
    
12. **Configured model and hyperparameter search**  
    To further improve the model, ridge and lasso regression estimators were set up and hyperparameter tuning was performed (e.g., via `GridSearchCV`) using cross-validation to find the best regularization strengths. The best model configuration discovered was re-trained on the training data.
    
13. **Model Performance Re-evaluation**  
    Regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root MSE (RMSE), and R² score were computed using `sklearn.metrics` to assess accuracy and fit.
    
14. **Re-visualization of the results**  
    Diagnostic plots (residual plots, feature importance or coefficients) were produced with `matplotlib` and `seaborn` to re-inspect model behavior and assumptions.
    
15. **Saved the final model**  
    The final trained model was serialized to disk using `joblib` for later reuse in inference or deployment.
