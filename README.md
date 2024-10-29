AI-Workbench
AI-Workbench is a no-code AI platform designed to simplify the machine learning workflow, making it accessible to users with minimal coding experience. This tool allows users to perform data preprocessing, model training, deployment, and experiment tracking, all within a user-friendly interface. AI-Workbench aims to democratize machine learning, providing comprehensive functionality for building and deploying AI solutions.

Key Features 
1. Data Import Options
Upload: Users can import data through multiple methods:
Excel/CSV Upload: Supports data import from local files or cloud storage in Excel or CSV format.
API Integration: Allows users to enter an API key to fetch data in JSON format, which is converted to a standard data frame.
Future Features: Plans include database connectivity using credentials and web-scraping capabilities for pulling data directly from web sources.


2. Data Preprocessing & Feature Engineering
Missing Value Handling:
0-Fill: Fills missing values with zero.
Forward Fill (Ffill) and Backward Fill (Bfill): Fill missing values based on adjacent data points.
Mean, Median, Min, and Max Fill: Replace missing values with specific aggregations.
Merge Options: Support for merging data with various join types (Inner, Outer, Left, Right) to facilitate combining datasets.
Outlier Detection:
IQR (Interquartile Range): Detects outliers based on the spread of the middle 50% of data.
Z-Score: Highlights values deviating significantly from the mean.
Data Imbalance Handling: Offers transformations to balance dataset classes as needed.


3. Data Exploration
One-Click Plotting: Provides instant visualizations for data insights.
Correlation Heatmaps: Visualize relationships between variables.
Pair Plots: Show interaction between variable pairs.
Manual Plotting: Customize plots with options like axis scaling, font size adjustments, and more, giving fine control over data visualization.
Time Series Analysis: Tools to analyze temporal data, including features like ACF, PACF, lag selection, and more.


4. Model Training
Supported Algorithms:
Linear Regression: For predicting continuous outcomes.
Random Forest: Ensemble method for classification and regression tasks.
Decision Tree: Simple yet effective for classification and prediction.
Support Vector Machine (SVM): Classification with optimal boundary creation.
XGBoost and Gradient Boosting: Ensemble models for high-accuracy predictive tasks.
K-Nearest Neighbors (KNN): Predicts based on similarity to nearest data points.
SGD Regressor: Iterative method for efficient prediction on large datasets.


5. Deployment & Experiment Tracking
Model Deployment: Deploy models with a few clicks. Model versions are managed, allowing for smooth and reproducible deployment.
MLflow Integration: Track and compare experiment metrics, model versions, and hyperparameters to optimize performance over time.
Tools & Technologies
Languages & Libraries: Python, Pandas, Sklearn
Frameworks: Streamlit for front-end, FastAPI for API management
Deployment & Containerization: Docker, BentoML for scalable deployments
Workflow Management: RabbitMQ and Celery for task scheduling
Experiment Tracking: MLflow for managing and visualizing machine learning experiments


Future Roadmap
Database Connectivity: Adding support for users to connect directly to databases for real-time querying.
Web Scraping: Enabling users to scrape web data directly within the platform.
Enhanced Visualization: Integrating advanced visualization options for model interpretability.


Getting Started
Prerequisites
Docker
Python 3.8+
Virtual environment for dependencies (recommended)
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/AI-Workbench.git
cd AI-Workbench
Set up the virtual environment and install dependencies:

bash
Copy code
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
Start Docker for containerized services like RabbitMQ.

Run the application:

bash
Copy code
streamlit run app.py
Usage
Upon launching the platform, users can:

Upload or import data.
Choose preprocessing and feature engineering steps.
Select a machine learning model and start training.
Track experiments and manage model deployments.
Contributing
Contributions are welcome! Please read the contribution guidelines and submit your pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
