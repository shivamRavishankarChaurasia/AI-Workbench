🚀 AI-Workbench
AI-Workbench is a powerful no-code AI platform that simplifies the machine learning workflow, making it accessible to users with minimal or no coding experience. It provides a complete suite of tools for data preprocessing, visualization, model training, deployment, and experiment tracking—all within a user-friendly interface.

Whether you're a data enthusiast, business analyst, or seasoned data scientist, AI-Workbench empowers you to build and deploy robust AI solutions with ease.

🌟 Key Features
📁 Data Import Options
Excel/CSV Upload: Import local files or from cloud storage.

API Integration: Fetch JSON data using API keys and convert to DataFrame.

Coming Soon:

Direct database connectivity using credentials.

Web scraping to pull data from websites dynamically.


🛠️ Data Preprocessing & Feature Engineering
Missing Value Handling:

0-Fill, Forward Fill (ffill), Backward Fill (bfill)

Mean, Median, Min, Max fill strategies

Data Merging:

Support for Inner, Outer, Left, and Right joins

Outlier Detection:

IQR method

Z-Score method

Data Imbalance Handling:

Apply transformations to balance dataset classes


📊 Data Exploration & Visualization
One-Click Plotting: Instantly visualize data insights

Manual Plotting: Customize axes, font sizes, etc.

Correlation Heatmaps and Pair Plots for deep insights

Time Series Analysis Tools:

ACF, PACF, and lag selection


🤖 Model Training
Supports various machine learning algorithms:

Linear Regression

Decision Tree

Random Forest

Support Vector Machine (SVM)

XGBoost & Gradient Boosting

K-Nearest Neighbors (KNN)

SGD Regressor


🚀 Deployment & Experiment Tracking
Model Deployment: One-click model deployment with version control

MLflow Integration: Track:

Experiment metrics

Model versions

Hyperparameters


🧰 Tech Stack
Area	Technologies Used
Languages & Libraries	Python, Pandas, Scikit-learn
Frameworks	Streamlit (Frontend), FastAPI (Backend API)
Deployment	Docker, BentoML
Task Scheduling	RabbitMQ, Celery
Experiment Tracking	MLflow

🔮 Future Roadmap
✅ Database Connectivity (MySQL, PostgreSQL, etc.)

⏳ Scheduling for online training 

# Clone the repository
git clone https://github.com/yourusername/AI-Workbench.git
cd AI-Workbench

# Set up a virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt







