Students Performance Analysis and Math Score Prediction
Project Overview
This project analyzes high school students' academic performance data to understand factors influencing math scores. Using Python, I performed exploratory data analysis, visualized relationships between variables, and built a machine learning model to predict math scores based on other features.

Key Objectives
Explore how variables like gender, race/ethnicity, parental education, lunch type, and test preparation affect math scores

Visualize data trends with heatmaps and boxplots for clear insights

Build and evaluate a linear regression model to predict math scores from available features

Data Source
Dataset containing students’ demographic information and their scores in math, reading, and writing.

Approach
Data loaded and cleaned using Pandas

Visualized feature correlations and score distributions using Seaborn and Matplotlib

Categorical variables encoded for model compatibility

Data split into training and testing sets (80/20 split)

Trained a Linear Regression model and evaluated using R² score and Mean Squared Error (MSE)

Saved predicted math scores to CSV for further analysis

Results
The model achieved an R² score of approximately 0.88, indicating a strong fit to the data

Mean Squared Error was around 28.28, showing prediction accuracy

Visualizations revealed clear trends, e.g., students who completed test preparation generally scored higher

Conclusion
This project demonstrates effective use of Python and machine learning for educational data analysis and prediction. It highlights the importance of test preparation and demographic factors on student performance. The project can be extended by experimenting with other algorithms or deeper feature analysis.

