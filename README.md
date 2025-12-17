# Fee Payment Default Prediction System

### Overview  
The Fee Payment Default Prediction System uses Machine Learning to identify students who are likely to delay fee payments or require payment reminders.  
It analyzes student fee and admission data such as course details, total fee, paid fee, and balance, then predicts payment behavior using trained ML models.

Features  
- Perform Exploratory Data Analysis (EDA) to understand fee payment patterns  
- Data preprocessing including encoding, scaling, and feature engineering  
- Handle class imbalance using SMOTE  
- Train and evaluate ML models like Logistic Regression, Decision Tree, and Random Forest  
- Model evaluation using accuracy, precision, recall, and F1-score  
- Business rule integration for reminder decisions  
- Final prediction & deployment using Streamlit web app  

## Tech Stack  
Language: Python  
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib  
Deployment: Streamlit  

## Project Structure  
Fee_Payment_Default_Prediction/
│  
├── data/  
│   ├── fee_payment_dataset.csv  
│  
├── notebooks/  
│   ├── 01_EDA.ipynb  
│   ├── 02_Preprocessing_and_Model_Training.ipynb  
│   ├── 03_Final_Model_and_Inference.ipynb  
│  
├── models/  
│   ├── delayed_payment_model.pkl  
│   ├── reminder_model.pkl  
│   ├── scaler.pkl  
│   ├── label_encoders.pkl  
│   ├── feature_columns.pkl  
│  
├── app/  
│   ├── app.py  
│  
├── utils/  
│   ├── preprocessing.py  
│   ├── evaluation.py  
│   ├── rule_engine.py  
│  
├── requirements.txt  
├── README.md  
├── LICENSE  
└── .gitignore  

### Installation  

## Step 1️⃣: Clone the repository  
bash
git clone https://github.com/your-username/Fee_Payment_Default_Prediction.git
cd Fee_Payment_Default_Prediction
Step 2️⃣: Install dependencies

bash
Copy code
pip install -r requirements.txt
Step 3️⃣: Run the Streamlit App

bash
Copy code
cd app
streamlit run app.py
Usage

Launch the Streamlit app

Enter student admission and fee details

Click on “Predict Payment Status”

View delayed payment and reminder prediction instantly

Example Prediction

Total Fee	Paid Fee	Balance	Delayed Payment	Needs Reminder
100000	60000	40000	Yes	Yes

Future Enhancements

SMS / Email reminder automation

ERP system integration

Cloud deployment (AWS / Streamlit Cloud)

Mobile application support

Student dropout prediction

## Author
Ankit Jadhav
Data Science & Machine Learning Enthusiast
www.linkedin.com/in/ankit-jadhav-5556ankit

## License
This project is open-source under the MIT License.
