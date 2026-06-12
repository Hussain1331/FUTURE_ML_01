# FUTURE_ML_01
WALMART SALES FORCASTING DASHBOARD

A machine learning project that predicts Walmart's weekly sales using historical sales data, store information, economic indicators, and promotional markdowns.
This project was built to understand the complete machine learning workflow — from data cleaning and preprocessing to model training, evaluation, and deployment using Streamlit.

 Project Overview
Retail businesses heavily rely on accurate sales forecasting for inventory management, staffing, and business planning.
In this project, I trained a machine learning model to predict weekly sales for Walmart stores based on various factors such as:
Store and Department information
Holiday weeks
Temperature
Fuel prices
CPI (Consumer Price Index)
Unemployment rate
Promotional markdowns

The trained model is integrated into an interactive Streamlit dashboard where users can input parameters and instantly generate sales predictions.

✨ Features
🔮 Sales Prediction

Predict weekly sales using a trained machine learning model.

📊 Interactive Dashboard
Clean and modern Streamlit interface with multiple analytics tabs.

📈 Trend Analysis
Visualize historical sales patterns and moving averages.

🏪 Store Analytics
Compare department performance and yearly trends.

🎯 Forecast Visualization
Future sales projection with confidence intervals.

⚡ Fast Performance

Uses Streamlit caching for faster model loading and data processing.

🛠️ Tech Stack

Programming Language
Python

Libraries
Pandas
NumPy
Scikit-Learn
Joblib
Plotly
Streamlit

Machine Learning
Regression-based sales forecasting model

📂 Project Workflow
1. Data Collection
Used the Walmart Sales Forecasting dataset containing:
Store details
Department information
Weekly sales
Economic indicators
Holiday information
Markdown promotions

2. Data Cleaning
Performed:
Missing value handling
Data type conversion
Date processing
Duplicate checking
Feature validation

3. Feature Engineering
Created additional features from the date column:
Year
Month
Week Number
Day
Converted categorical values into machine-readable formats for training.

4. Model Training
The dataset was split into training and testing sets.
The model was trained to learn relationships between business factors and weekly sales.
Model performance was evaluated using regression metrics before deployment.

5. Deployment
The trained model was saved using Joblib and deployed through a Streamlit dashboard.
Users can enter custom inputs and get instant sales predictions.

📊 Input Features
Feature	Description
Store	Walmart Store ID
Dept	Department ID
IsHoliday	Holiday Week Indicator
Temperature	Temperature in °F
Fuel Price	Fuel Cost
MarkDown1-5	Promotional Discounts
CPI	Consumer Price Index
Unemployment	Unemployment Rate
Year	Year
Month	Month
Week	Week Number
Day	Day

🖥️ Dashboard Sections
Forecast
Generates sales predictions and forecast visualizations.
Trends
Displays weekly sales history and seasonal patterns.
Store View
Provides department-level and store-level insights.


Dashboard Home
Sales Prediction Result
Trend Analysis
Store Analytics



🎓 What I Learned
Through this project, I gained hands-on experience in:
Data Cleaning
Exploratory Data Analysis (EDA)
Feature Engineering
Machine Learning Model Training
Model Evaluation
Streamlit Dashboard Development
Data Visualization
Model Deployment
