# Student Performance Prediction Project

This project predicts student performance based on demographic and academic data. It includes **regression models** for predicting final grades, **classification models** for pass/fail prediction, and **clustering** for grouping students.

---

## 📂 Dataset

The dataset used is `StudentsPerformance.csv`, which contains:

| Column                       | Type       |
|-------------------------------|-----------|
| gender                        | categorical |
| race/ethnicity                | categorical |
| parental level of education   | categorical |
| lunch                         | categorical |
| test preparation course       | categorical |
| math score                    | numeric   |
| reading score                 | numeric   |
| writing score                 | numeric   |

- `G3` (Final grade) is calculated as the **average of math, reading, and writing scores**.
- Classification target `pass` is created: `1` if `G3 >= 50%`, else `0`.

---

## ⚙️ Features & Preprocessing

- Label encoding for categorical features  
- Feature scaling with `StandardScaler`  
- Train/test split: 80/20  

---

## 🤖 Models Trained

### Regression Models (predict final grade `G3`)
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

### Classification Models (Pass/Fail)
- Logistic Regression  
- Random Forest Classifier  

### Evaluation Metrics
- MAE, RMSE, R² (Regression)  
- Accuracy, Confusion Matrix, ROC Curve (Classification)  

### Clustering
- KMeans clustering to group students into 3 categories  

---

## 📊 Visualizations
- Correlation heatmap  
- Boxplots & histograms of grades  
- Confusion matrix & ROC curve  
- Scatter plot of student clusters  

---

## 💻 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/StudentPerformanceProject.git
cd StudentPerformanceProject