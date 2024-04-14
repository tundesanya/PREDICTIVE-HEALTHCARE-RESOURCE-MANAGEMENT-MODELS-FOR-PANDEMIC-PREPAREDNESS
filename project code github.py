
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd

# Read the data
data = pd.read_csv('COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv')
data = data.iloc[:30, :100]

# Select the first 30 columns
datas = data.iloc[:30, :100]

# View first and last 5 observations
print(data)

columns_list = data.columns.tolist()

# Print the list of columns
print("List of Columns:")
print(columns_list)

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your dataset
correlation_matrix = data.iloc[:, 3:40].corr()

# Display the correlation matrix
print("Correlation Matrix:")
#print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=False)
plt.title('Correlation Heatmap')
plt.show()

# Describe statistical information of data
print(data.describe())
# Below stats show that 75 percentile of obseravtions belong to class 1 

# Look for missing values
print(data.isnull().sum())     

# Replace null or empty values with zero
data.fillna(0, inplace=True)

# Display the count of null values after replacement
print("\nNull Values After Replacement:")
print(data.isnull().sum())

# You might need to preprocess the 'date' column and set it as the index for time series modeling
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Feature selection
features = data[['critical_staffing_shortage_today_yes', 'inpatient_beds', 'staffed_adult_icu_bed_occupancy', 'staffed_icu_adult_patients_confirmed_covid_coverage', 'total_adult_patients_hospitalized_confirmed_and_suspected_covid', 'total_adult_patients_hospitalized_confirmed_and_suspected_covid_coverage', 'total_adult_patients_hospitalized_confirmed_covid', 'total_adult_patients_hospitalized_confirmed_covid_coverage', 'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid', 'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid_coverage', 'total_pediatric_patients_hospitalized_confirmed_covid', 'total_pediatric_patients_hospitalized_confirmed_covid_coverage', 'total_staffed_adult_icu_beds', 'total_staffed_adult_icu_beds_coverage']]
target = data['total_adult_patients_hospitalized_confirmed_covid']



from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

from sklearn.svm import SVR
# SVM Model
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Prophet Time Series Model
prophet_data = pd.DataFrame({'ds': data.index, 'y': data['total_adult_patients_hospitalized_confirmed_covid']})
prophet_model = Prophet()
prophet_model.fit(prophet_data)

# Create a future dataframe for predictions
future = prophet_model.make_future_dataframe(periods=len(X_test))

# Make predictions
prophet_predictions = prophet_model.predict(future)

# Evaluate models using Mean Absolute Error
rf_mae = mean_absolute_error(y_test, rf_predictions)
svm_mae = mean_absolute_error(y_test, svm_predictions)
prophet_mae = mean_absolute_error(y_test, prophet_predictions['yhat'][:len(y_test)])

print(f"Random Forest MAE: {rf_mae}")
print(f"SVM MAE: {svm_mae}")
print(f"Prophet MAE: {prophet_mae}")

from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

# Evaluate models using Mean Absolute Error and Classification Metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
svm_mae = mean_absolute_error(y_test, svm_predictions)
prophet_mae = mean_absolute_error(y_test, prophet_predictions['yhat'][:len(y_test)])

# Binary classification for simplicity
y_test_binary = (y_test > y_test.mean()).astype(int)
rf_predictions_binary = (rf_predictions > rf_predictions.mean()).astype(int)
svm_predictions_binary = (svm_predictions > svm_predictions.mean()).astype(int)
prophet_predictions_binary = (prophet_predictions['yhat'][:len(y_test)] > prophet_predictions['yhat'][:len(y_test)].mean()).astype(int)

# Classification Metrics
rf_accuracy = accuracy_score(y_test_binary, rf_predictions_binary)
rf_precision = precision_score(y_test_binary, rf_predictions_binary)
rf_recall = recall_score(y_test_binary, rf_predictions_binary)
rf_f1 = f1_score(y_test_binary, rf_predictions_binary)

svm_accuracy = accuracy_score(y_test_binary, svm_predictions_binary)
svm_precision = precision_score(y_test_binary, svm_predictions_binary)
svm_recall = recall_score(y_test_binary, svm_predictions_binary)
svm_f1 = f1_score(y_test_binary, svm_predictions_binary)

print(f"Random Forest MAE: {rf_mae}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1 Score: {rf_f1}")

print(f"SVM MAE: {svm_mae}")
print(f"SVM Accuracy: {svm_accuracy}")
print(f"SVM Precision: {svm_precision}")
print(f"SVM Recall: {svm_recall}")
print(f"SVM F1 Score: {svm_f1}")



from sklearn.metrics import confusion_matrix
import seaborn as sns

# Feature selection
features = data.iloc[:, :50]  # Select the first 50 features
target = data['total_adult_patients_hospitalized_confirmed_covid']

# Display confusion matrix and correlation matrix
def evaluate_models(true_values, predicted_values, model_name):
    # Confusion Matrix
    cm = confusion_matrix(true_values, predicted_values)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Correlation Matrix
    correlation_matrix = np.corrcoef(true_values, predicted_values)
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=False)
    plt.title(f'{model_name} Correlation Matrix')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.show()

# Evaluate models using Mean Absolute Error and Classification Metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
svm_mae = mean_absolute_error(y_test, svm_predictions)
prophet_mae = mean_absolute_error(y_test, prophet_predictions['yhat'][:len(y_test)])

# Binary classification for simplicity
y_test_binary = (y_test > y_test.mean()).astype(int)
rf_predictions_binary = (rf_predictions > rf_predictions.mean()).astype(int)
svm_predictions_binary = (svm_predictions > svm_predictions.mean()).astype(int)
prophet_predictions_binary = (prophet_predictions['yhat'][:len(y_test)] > prophet_predictions['yhat'][:len(y_test)].mean()).astype(int)

# Display confusion matrix and correlation matrix for each model
evaluate_models(y_test_binary, rf_predictions_binary, 'Random Forest')
evaluate_models(y_test_binary, svm_predictions_binary, 'SVM')
evaluate_models(y_test_binary, prophet_predictions_binary, 'Prophet')





# Scatter plot for Random Forest with lines
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, alpha=0.5, label='Scatter Plot')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.title('Random Forest Model - True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for SVM with lines
plt.figure(figsize=(8, 6))
plt.scatter(y_test, svm_predictions, alpha=0.5, label='Scatter Plot')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.title('SVM Model - True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for Prophet with lines
plt.figure(figsize=(8, 6))
plt.scatter(y_test, prophet_predictions['yhat'][:len(y_test)], alpha=0.5, label='Scatter Plot')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
plt.title('Prophet Model - True vs Predicted')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()




import seaborn as sns

# ... (previous code)

# Density plot for Random Forest
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test, label='True Values', shade=True)
sns.kdeplot(rf_predictions, label='Random Forest Predictions', shade=True)
plt.title('Random Forest Model - Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Density plot for SVM
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test, label='True Values', shade=True)
sns.kdeplot(svm_predictions, label='SVM Predictions', shade=True)
plt.title('SVM Model - Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Density plot for Prophet
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test, label='True Values', shade=True)
sns.kdeplot(prophet_predictions['yhat'][:len(y_test)], label='Prophet Predictions', shade=True)
plt.title('Prophet Model - Density Plot')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

