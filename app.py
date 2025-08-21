import streamlit as st
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import altair as alt
import io

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

# ---------- Convert to DataFrame ----------
if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    df['date_only'] = df['date'].dt.date
    df['month'] = df['date'].dt.to_period('M').astype(str)

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters & Budgets")
if st.session_state.expenses:
    months = sorted(df['month'].unique())
    categories = sorted(df['category'].unique())
    
    selected_month = st.sidebar.selectbox("Select Month", ["All"] + months)
    selected_category = st.sidebar.selectbox("Select Category", ["All"] + categories)
    
    # Budget inputs
    st.sidebar.subheader("Set Budget per Category")
    budgets = {}
    for cat in categories:
        budgets[cat] = st.sidebar.number_input(f"{cat} Budget", min_value=0.0, step=10.0)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_month != "All":
        filtered_df = filtered_df[filtered_df['month'] == selected_month]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
else:
    filtered_df = pd.DataFrame()
    budgets = {}

# ---------- Show Expenses ----------
st.header("All Expenses")
if not filtered_df.empty:
    total_amount = filtered_df['amount'].sum()
    st.subheader(f"Total Expenses: ₹{total_amount:.2f}")
    for _, exp in filtered_df.iterrows():
        st.write(f"{exp['description']} - ₹{exp['amount']} ({exp['category']})")
else:
    st.write("No expenses to show for selected filters.")

# ---------- Category Analytics ----------
st.header("Category Analytics")
if not filtered_df.empty:
    chart_data = filtered_df.groupby('category')['amount'].sum().reset_index()
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='category', y='amount', color='category'
    )
    st.altair_chart(chart, use_container_width=True)

# ---------- Monthly Breakdown ----------
st.header("Monthly Breakdown")
if st.session_state.expenses:
    month_data = df.groupby('month')['amount'].sum().reset_index()
    month_chart = alt.Chart(month_data).mark_bar().encode(
        x='month', y='amount', color='month'
    )
    st.altair_chart(month_chart, use_container_width=True)

# ---------- Next Month Expense Prediction ----------
st.header("Next Month Expense Prediction")
if not filtered_df.empty:
    daily_totals = filtered_df.groupby('date_only')['amount'].sum().values
    avg_daily = daily_totals.mean()
    next_month_pred = avg_daily * 30  # Predict for 30 days
    st.write(f"Predicted total expense for next month: ₹{next_month_pred:.2f}")
else:
    st.write("No data to predict next month expense.")

# ---------- Budget Alerts ----------
st.header("Budget Alerts")
if budgets and not filtered_df.empty:
    spent_per_category = filtered_df.groupby('category')['amount'].sum()
    alerts = []
    for cat, budget in budgets.items():
        spent = spent_per_category.get(cat, 0)
        if spent > budget > 0:
            alerts.append(f"⚠️ {cat} exceeded budget! Spent ₹{spent:.2f} / Budget ₹{budget:.2f}")
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("No budgets exceeded!")
else:
    st.write("Set budgets to see alerts.")

# ---------- Download Report ----------
st.header("Download Report")
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f'expense_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )
else:
    st.write("No data available to download.")
