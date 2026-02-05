import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Library Management System",
    page_icon="📚",
    layout="centered"
)

# -------------------------------
# Background Styling (CSS)
# -------------------------------
def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
            background-attachment: fixed;
        }

        h1, h2, h3, h4 {
            color: #2c3e50;
        }

        .stSidebar {
            background-color: #2c3e50;
        }

        .stSidebar h2, .stSidebar label {
            color: white;
        }

        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-size: 16px;
        }

        .stButton>button:hover {
            background-color: #2980b9;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_background()

# -------------------------------
# Title
# -------------------------------
st.title("📚 Library Management System")
st.subheader("Frequent Library User Prediction")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("library_data_100.csv")

data = load_data()

st.write("### 📄 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Encoding Target Column
# -------------------------------
le = LabelEncoder()
data["FrequentUser"] = le.fit_transform(data["FrequentUser"])
# Yes -> 1, No -> 0

# -------------------------------
# Features & Target
# -------------------------------
X = data[["StudentAge", "BooksIssued", "LateReturns", "MembershipYears"]]
y = data["FrequentUser"]

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧑‍🎓 Enter Student Details")

age = st.sidebar.number_input("Student Age", min_value=10, max_value=60, value=20)
books = st.sidebar.number_input("Books Issued", min_value=0, max_value=50, value=5)
late = st.sidebar.number_input("Late Returns", min_value=0, max_value=20, value=1)
membership = st.sidebar.number_input("Membership Years", min_value=0, max_value=10, value=1)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("🔍 Predict"):
    input_data = [[age, books, late, membership]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediction: Frequent Library User")
    else:
        st.error("❌ Prediction: Not a Frequent Library User")

# -------------------------------
# Model Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)
st.write(f"### 📊 Model Accuracy: **{accuracy * 100:.2f}%**")
