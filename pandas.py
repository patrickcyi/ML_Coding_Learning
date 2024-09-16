# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV file and handle datetime format
data = pd.read_csv('path_to_your_file.csv')

# If there is a datetime column
if 'date_column' in data.columns:
    data['date_column'] = pd.to_datetime(data['date_column'])
    # You can extract year, month, or day as features if necessary
    data['year'] = data['date_column'].dt.year
    data['month'] = data['date_column'].dt.month
    data['day'] = data['date_column'].dt.day

# Step 2: Feature selection (check correlations and plot)
# If it's a regression problem, use numerical features' correlations
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# Select features based on correlation or feature importance from a model
# Assume 'A' is the target column
X = data.drop(columns=['A'])  # Features
y = data['A']  # Target

# Optional: If there are missing values, you can handle them
X = X.fillna(X.mean())  # Example of filling missing values

# Step 3: StandardScaler for numerical features
# OneHotEncoder for categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model training pipeline (RandomForest as an example)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)



# Optional: Feature importance visualization for RandomForest
importances = model.named_steps['classifier'].feature_importances_
feature_names = numerical_features.tolist() + model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features).tolist()

# Plot the feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
