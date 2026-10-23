
import streamlit as st
from dotenv import load_dotenv
from google import genai
import os
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Expense Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# ENVIRONMENT & GEMINI SETUP
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "expenses" not in st.session_state:
    st.session_state.expenses = []

if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = ""

if "budget" not in st.session_state:
    st.session_state.budget = 0.0

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .app-title {
        text-align: center;
        font-size: 38px;
        font-weight: 750;
        color: #172033;
        margin-bottom: 4px;
    }

    .app-subtitle {
        text-align: center;
        font-size: 16px;
        color: #667085;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-heading {
        font-size: 20px;
        font-weight: 650;
        color: #172033;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    /* Summary cards */
    .summary-card {
        background: #f8fafc;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    .summary-label {
        font-size: 13px;
        color: #667085;
        margin-bottom: 7px;
    }

    .summary-value {
        font-size: 22px;
        font-weight: 750;
        color: #172033;
        word-break: break-word;
    }

    /* Budget cards */
    .budget-card {
        background: #f8fafc;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    .budget-title {
        font-size: 14px;
        color: #667085;
        margin-bottom: 6px;
    }

    .budget-value {
        font-size: 25px;
        font-weight: 750;
        color: #172033;
    }

    /* AI insights */
    .insight-card {
        background: #f8fafc;
        border: 1px solid #e4e7ec;
        border-left: 4px solid #4f46e5;
        border-radius: 14px;
        padding: 22px;
        line-height: 1.7;
        color: #273142;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    /* Distribution */
    .distribution-card {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 10px;
    }

    .distribution-name {
        font-size: 14px;
        font-weight: 600;
        color: #344054;
    }

    .distribution-value {
        font-size: 13px;
        color: #667085;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 55px 20px;
        color: #667085;
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 14px;
        margin-top: 15px;
    }

    .empty-title {
        font-size: 20px;
        font-weight: 650;
        color: #344054;
        margin-bottom: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e4e7ec;
    }

    /* Mobile responsive */
    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.2rem;
        }

        .app-title {
            font-size: 28px;
        }

        .app-subtitle {
            font-size: 14px;
        }

        .section-heading {
            font-size: 18px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 💰 Expense Analyzer")

    st.caption("A simple way to understand your spending.")

    st.divider()

    st.markdown("### Quick Stats")

    if st.session_state.expenses:

        sidebar_df = pd.DataFrame(st.session_state.expenses)

        sidebar_total = sidebar_df["Amount"].sum()
        sidebar_count = len(sidebar_df)

        st.metric(
            "Total Spending",
            f"Rs. {sidebar_total:,.0f}"
        )

        st.metric(
            "Expenses",
            sidebar_count
        )

        if st.session_state.budget > 0:

            remaining = st.session_state.budget - sidebar_total

            st.metric(
                "Budget Remaining",
                f"Rs. {remaining:,.0f}"
            )

    else:

        st.info("Add your first expense to see your statistics.")

    st.divider()

    st.markdown("### Monthly Budget")

    budget_input = st.number_input(
        "Set your budget (Rs.)",
        min_value=0.0,
        value=st.session_state.budget,
        step=1000.0
    )

    st.session_state.budget = budget_input

    st.divider()

    st.markdown("### ℹ️ About")

    st.caption(
        "AI Expense Analyzer helps you track expenses, "
        "understand spending patterns and receive practical "
        "AI-powered saving suggestions."
    )

    st.caption("Built with Python, Streamlit, Pandas & Gemini AI.")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="app-title"> AI Expense Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Understand your spending. Make smarter financial decisions.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# ADD EXPENSE
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">➕ Add an Expense</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        category = st.selectbox(
            "Category",
            [
                "Food",
                "Transport",
                "Education",
                "Shopping",
                "Bills",
                "Entertainment",
                "Health",
                "Other"
            ]
        )

    with col2:

        amount = st.number_input(
            "Amount (Rs.)",
            min_value=0.0,
            step=100.0
        )

    col3, col4 = st.columns(2)

    with col3:

        expense_date = st.date_input(
            "Date"
        )

    with col4:

        description = st.text_input(
            "Description",
            placeholder="e.g. Lunch, bus fare, books..."
        )

    add_clicked = st.button(
        "Add Expense",
        use_container_width=True
    )

    if add_clicked:

        if amount > 0:

            st.session_state.expenses.append(
                {
                    "Date": expense_date,
                    "Category": category,
                    "Description": description if description else "-",
                    "Amount": amount
                }
            )

            # Clear old AI analysis after adding new data
            st.session_state.ai_insights = ""

            st.success("Expense added successfully.")

        else:

            st.warning(
                "Please enter an amount greater than 0."
            )

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if st.session_state.expenses:

    df = pd.DataFrame(st.session_state.expenses)

    total = df["Amount"].sum()
    count = len(df)
    average = total / count

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_category = category_totals.idxmax()

    # --------------------------------------------------
    # BUDGET OVERVIEW
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">💰 Budget Overview</div>',
        unsafe_allow_html=True
    )

    if st.session_state.budget > 0:

        budget = st.session_state.budget
        remaining = budget - total

        budget_col1, budget_col2, budget_col3 = st.columns(3)

        with budget_col1:

            st.markdown(
                f"""
                <div class="budget-card">
                    <div class="budget-title">Monthly Budget</div>
                    <div class="budget-value">
                        Rs. {budget:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with budget_col2:

            st.markdown(
                f"""
                <div class="budget-card">
                    <div class="budget-title">Spent</div>
                    <div class="budget-value">
                        Rs. {total:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with budget_col3:

            remaining_label = "Remaining"

            if remaining < 0:
                remaining_label = "Over Budget"

            st.markdown(
                f"""
                <div class="budget-card">
                    <div class="budget-title">{remaining_label}</div>
                    <div class="budget-value">
                        Rs. {abs(remaining):,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        percentage_used = min(
            total / budget,
            1.0
        )

        st.progress(
            percentage_used,
            text=f"{(total / budget) * 100:.1f}% of your budget used"
        )

        if remaining < 0:

            st.error(
                f"You are Rs. {abs(remaining):,.0f} over your budget."
            )

        elif remaining <= budget * 0.2:

            st.warning(
                "You're getting close to your monthly budget."
            )

        else:

            st.success(
                f"You have Rs. {remaining:,.0f} left in your budget."
            )

    else:

        st.info(
            "Set a monthly budget from the sidebar to track "
            "your remaining balance."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">📊 Dashboard Summary</div>',
        unsafe_allow_html=True
    )

    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Total Spending</div>
                <div class="summary-value">
                    Rs. {total:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col2:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">No. of Expenses</div>
                <div class="summary-value">
                    {count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col3:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Top Category</div>
                <div class="summary-value">
                    {highest_category}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col4:

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Average Expense</div>
                <div class="summary-value">
                    Rs. {average:,.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.markdown(
            '<div class="section-heading">'
            'Spending by Category'
            '</div>',
            unsafe_allow_html=True
        )

        st.bar_chart(
            category_totals,
            use_container_width=True
        )

    with chart_col2:

        st.markdown(
            '<div class="section-heading">'
            '📈 Spending Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        for category_name, category_amount in category_totals.items():

            percentage = (
                category_amount / total
            ) * 100

            st.markdown(
                f"""
                <div class="distribution-card">
                    <div style="display:flex;
                                justify-content:space-between;">
                        <span class="distribution-name">
                            {category_name}
                        </span>
                        <span class="distribution-value">
                            Rs. {category_amount:,.0f}
                            ({percentage:.1f}%)
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                percentage / 100
            )

    # --------------------------------------------------
    # EXPENSE TABLE
    # --------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">🧾 Your Expenses</div>',
        unsafe_allow_html=True
    )

    display_df = df.copy()

    display_df["Amount"] = display_df[
        "Amount"
    ].apply(
        lambda x: f"Rs. {x:,.2f}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # DELETE EXPENSE
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">🗑️ Manage Expenses</div>',
        unsafe_allow_html=True
    )

    delete_col1, delete_col2 = st.columns(2)

    with delete_col1:

        expense_options = []

        for index, expense in enumerate(
            st.session_state.expenses
        ):

            expense_options.append(
                f"{index + 1}. "
                f"{expense['Category']} - "
                f"Rs. {expense['Amount']:,.0f} - "
                f"{expense['Date']}"
            )

        selected_expense = st.selectbox(
            "Select an expense to delete",
            expense_options
        )

        if st.button(
            "Delete Selected Expense",
            use_container_width=True
        ):

            selected_index = (
                expense_options.index(
                    selected_expense
                )
            )

            del st.session_state.expenses[
                selected_index
            ]

            st.session_state.ai_insights = ""

            st.rerun()

    with delete_col2:

        st.write("")

        st.write("")

        if st.button(
            "Clear All Expenses",
            use_container_width=True
        ):

            st.session_state.expenses = []
            st.session_state.ai_insights = ""

            st.rerun()

    # --------------------------------------------------
    # DOWNLOAD CSV
    # --------------------------------------------------

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Expenses as CSV",
        data=csv_data,
        file_name="my_expenses.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # --------------------------------------------------
    # AI INSIGHTS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-heading">'
        '🤖 Your Spending Insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Get practical suggestions based on your current spending."
    )

    analyze_clicked = st.button(
        " Analyze My Spending",
        use_container_width=True
    )

    if analyze_clicked:

        expense_text = df.to_string(
            index=False
        )

        budget_information = (
            f"Monthly budget: Rs. {st.session_state.budget:,.2f}"
            if st.session_state.budget > 0
            else "No monthly budget was set."
        )

        prompt = f"""
You are a friendly personal finance assistant helping a student.

Analyze the following expense data:

{expense_text}

Total spending: Rs. {total:,.2f}

{budget_information}

Provide a short, practical and easy-to-understand analysis.

Include these sections:

### Spending Summary
Briefly explain the overall spending pattern.

### Highest Spending
Identify the category where the user spent the most.

### Areas to Reduce
Mention realistic areas where spending could be reduced.

### Saving Tips
Give exactly 3 practical saving tips.

If a monthly budget is available, also comment briefly on whether
the user is within or over the budget.

Use simple English.
Be supportive and non-judgmental.
Do not make unrealistic financial claims.
Avoid complicated financial terminology.
"""

        with st.spinner(
            "Analyzing your spending..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                st.session_state.ai_insights = response.text

            except Exception as e:

                st.error(
                    "Unable to generate AI insights right now. "
                    "Please check your Gemini API connection."
                )

    # --------------------------------------------------
    # SHOW SAVED AI INSIGHTS
    # --------------------------------------------------

    if st.session_state.ai_insights:

        st.markdown(
            '<div class="insight-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            st.session_state.ai_insights
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

else:

    # --------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------

 st.markdown("""
<div class="empty-state">
    <h3>Your spending dashboard is waiting for you.</h3>
    <p>Add your first expense above to start understanding your spending.</p>
</div>
""", unsafe_allow_html=True)
