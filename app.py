import streamlit as st
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import altair as alt

st.set_page_config(page_title="AI Expense Tracker")

# --------- In-memory database ----------
if 'expenses' not in st.session_state:
    st.session_state.expenses = []

# ---------- ML Model Setup ----------
train_desc = [
    "lunch at cafe", "train ticket", "electricity bill", "movie tickets", "groceries from store",
    "bus fare", "restaurant dinner", "internet bill", "clothes shopping", "taxi ride"
]
train_labels = ["Food","Travel","Bills","Entertainment","Food","Travel","Food","Bills","Shopping","Travel"]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_desc)
model = LogisticRegression()
model.fit(X_train, train_labels)

def predict_category(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

# ---------- Add Manual Expense ----------
st.header("Add Manual Expense")
with st.form("expense_form"):
    desc = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    submitted = st.form_submit_button("Add Expense")
    if submitted:
        category = predict_category(desc)
        st.session_state.expenses.append({
            "description": desc,
            "amount": amount,
            "category": category,
            "date": datetime.now()
        })
        st.success(f"Added: {desc} - ₹{amount} ({category})")

# ---------- Show Expenses ----------
st.header("All Expenses")
for exp in st.session_state.expenses:
    st.write(f"{exp['description']} - ₹{exp['amount']} ({exp['category']})")

# ---------- Analytics ----------
st.header("Analytics")
if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    chart_data = df.groupby('category')['amount'].sum().reset_index()
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='category', y='amount', color='category'
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- Future Expense Prediction ----------
st.header("Next Day Expense Prediction")
if st.session_state.expenses:
    amounts = np.array([e['amount'] for e in st.session_state.expenses]).reshape(-1,1)
    days = np.arange(len(st.session_state.expenses)).reshape(-1,1)
    model_pred = LinearRegression()
    model_pred.fit(days, amounts)
    pred = model_pred.predict(np.array([[len(st.session_state.expenses)]]))[0][0]
    st.write(f"Predicted next expense: ₹{pred:.2f}")
