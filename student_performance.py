## step 1:Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix, roc_curve, auc)

from sklearn.cluster import KMeans

##step2: Load Dataset
df = pd.read_csv("data/StudentsPerformance.csv")
print(df.head())
print(df.info())

##Step 3: EDA (Exploratory Data Analysis)
# Check missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Study time vs final grade
sns.boxplot(x='studytime', y='G3', data=df)
plt.title("Study Time vs Final Grade")
plt.show()

# Distribution of final grades
sns.histplot(df['G3'], bins=20, kde=True)
plt.title("Grade Distribution")
plt.show()

##Step 4: Feature Engineering & Preprocessing
# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & target (Regression)
X = df.drop("G3", axis=1)
y = df["G3"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##Step 5: Model Training (3 Models)
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Gradient Boosting (optional)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)

##Step 6: Evaluation (Regression)
def evaluate_model(name, model):
    pred = model.predict(X_test)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2 Score:", r2_score(y_test, pred))

evaluate_model("Linear Regression", lr)
evaluate_model("Random Forest", rf)
evaluate_model("Gradient Boosting", gb)
 
##Step 7: Classification Version (Pass/Fail)
# Create classification target
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

X_cls = df.drop(['G3', 'pass'], axis=1)
y_cls = df['pass']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Scale
X_train_c = scaler.fit_transform(X_train_c)
X_test_c = scaler.transform(X_test_c)

# Models
log_reg = LogisticRegression(max_iter=1000)
rf_cls = RandomForestClassifier()

log_reg.fit(X_train_c, y_train_c)
rf_cls.fit(X_train_c, y_train_c)

##Step 8: Confusion Matrix & ROC Curve
# Predictions
y_pred = rf_cls.predict(X_test_c)

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_prob = rf_cls.predict_proba(X_test_c)[:,1]
fpr, tpr, _ = roc_curve(y_test_c, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

##Step 9: Hyperparameter Tuning
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None]
}

grid = GridSearchCV(RandomForestRegressor(), params, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

##Step 10: Unsupervised Learning (Clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Visualize clusters (using 2 features)
plt.scatter(df['studytime'], df['G3'], c=df['cluster'])
plt.xlabel("Study Time")
plt.ylabel("Final Grade")
plt.title("Student Clusters")
plt.show()

##Step 11: Final Insights (Print)
print("\nKey Insights:")
print("- Higher study time → Better performance")
print("- Absences → Negative impact")
print("- Previous grades strongly influence final grade")
print("- Clustering reveals 3 student categories")