import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("StudentsPerformance.csv")

# Quick look at data
print("First 5 rows of the dataset:")
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# Show correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
# Compare math scores based on test preparation course
plt.figure(figsize=(8,6))
sns.boxplot(x='test preparation course', y='math score', data=df, palette='Set2')
plt.title("Math Score vs Test Preparation Course")
plt.xlabel("Test Preparation Course")
plt.ylabel("Math Score")
plt.show()
# Compare math scores based on gender
plt.figure(figsize=(8,6))
sns.boxplot(x='gender', y='math score', data=df, palette='Set1')
plt.title("Math Score vs Gender")
plt.xlabel("Gender")
plt.ylabel("Math Score")
plt.show()

# Convert categorical columns to numeric
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['race/ethnicity'] = df['race/ethnicity'].astype('category').cat.codes
df['parental level of education'] = df['parental level of education'].astype('category').cat.codes
df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})
df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})

# Choose features (X) and target (y)
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
        'reading score', 'writing score']]
y = df['math score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict math scores for test data
y_pred = model.predict(X_test)

# Evaluate model performance
print(f"\nR2 score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Create a DataFrame to compare Actual vs Predicted scores
results = pd.DataFrame({
    'Actual Math Score': y_test,
    'Predicted Math Score': y_pred
})

# Save the results to a CSV file
results.to_csv("math_score_predictions.csv", index=False)
print("\n✅ Predictions saved to 'math_score_predictions.csv'")
"""
Students Performance Analysis and Math Score Prediction

Project Overview:
This project analyzes high school students' academic performance data to understand factors influencing math scores.
Using Python, I performed exploratory data analysis, visualized relationships between variables, and built a machine
learning model to predict math scores based on other features.

Key Objectives:
- Explore how variables like gender, race/ethnicity, parental education, lunch type, and test preparation affect math scores
- Visualize data trends with heatmaps and boxplots for clear insights
- Build and evaluate a linear regression model to predict math scores from available features

Results:
- The model achieved an R2 score of approximately 0.88, indicating a strong fit to the data
- Visualizations revealed clear trends, e.g., students who completed test preparation generally scored higher
"""
print("\n--- Project Summary ---")
print("Students Performance Analysis and Math Score Prediction")
print("This project analyzes high school students' academic performance data to understand factors influencing math scores.")
print("Key objectives include exploring demographic impacts, visualizing data, and predicting math scores using Linear Regression.")
print("The model achieved an R2 score of around 0.88, showing good prediction accuracy.")
print("Visualizations highlight trends such as the positive effect of test preparation courses.\n")
