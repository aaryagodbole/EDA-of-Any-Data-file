import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Complete Auto EDA", page_icon="📊", layout="wide")
st.title("📊 Complete Exploratory Data Analysis")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Upload File
file = st.file_uploader("📂 Upload a file", type=["csv", "xlsx", "xls"])

if file is not None:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file, encoding="ISO-8859-1")
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    st.success("✅ File loaded successfully!")

    # Dataset Overview
    st.subheader("🔍 Dataset Overview")
    st.dataframe(df.sample(min(len(df), 100)), use_container_width=True)
    st.caption("Showing a sample of 100 rows")
    st.markdown(f"**Shape:** {df.shape}")
    st.markdown(f"**Columns:** {list(df.columns)}")
    st.write("**Data Types:**")
    st.write(df.dtypes)

    # Date Column Detection
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    if not date_cols:
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue

    # Date Range Filtering
    if date_cols:
        st.subheader("📅 Date Filter")
        date_col = st.selectbox("Select Date Column", date_cols)
        start = st.date_input("Start Date", df[date_col].min())
        end = st.date_input("End Date", df[date_col].max())
        df = df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]

    # Sidebar Filters for Categorical
    st.sidebar.header("🔎 Filter Data")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        selected = st.sidebar.multiselect(f"Filter by {col}", df[col].dropna().unique()[:100])
        if selected:
            df = df[df[col].isin(selected)]

    # Summary Statistics (Lazy load)
    with st.expander("📊 Summary Statistics"):
        st.dataframe(df.describe(include="all"))

    with st.expander("❗ Missing Values"):
        st.dataframe(df.isnull().sum())

    # Numerical & Categorical Columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Chart Rendering Control
    st.header("📈 Visualizations")
    if st.checkbox("✅ Show Visualizations", value=False):

        # Bar Chart
        if cat_cols and num_cols:
            with st.expander("🟦 Bar Chart"):
                bar_x = st.selectbox("Bar X (Categorical)", cat_cols, key="barx")
                bar_y = st.selectbox("Bar Y (Numeric)", num_cols, key="bary")
                fig_bar = px.bar(df.groupby(bar_x)[bar_y].sum().reset_index(), x=bar_x, y=bar_y, template="seaborn")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Line Chart
        if date_cols and num_cols:
            with st.expander("📉 Line Chart"):
                df["Month"] = df[date_col].dt.to_period("M").astype(str)
                line_y = st.selectbox("Line Y (Numeric)", num_cols, key="liney")
                fig_line = px.line(df.groupby("Month")[line_y].sum().reset_index(), x="Month", y=line_y, markers=True)
                st.plotly_chart(fig_line, use_container_width=True)

        # Pie Chart
        if cat_cols and num_cols:
            with st.expander("🥧 Pie Chart"):
                pie_col = st.selectbox("Pie Category", cat_cols, key="piecol")
                pie_val = st.selectbox("Pie Value", num_cols, key="pieval")
                fig_pie = px.pie(df, names=pie_col, values=pie_val, hole=0.5)
                st.plotly_chart(fig_pie, use_container_width=True)

        # Scatter Plot
        if len(num_cols) >= 2:
            with st.expander("📍 Scatter Plot"):
                scatter_x = st.selectbox("X Axis", num_cols, key="scatterx")
                scatter_y = st.selectbox("Y Axis", num_cols, index=1, key="scattery")
                fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color=cat_cols[0] if cat_cols else None, size=num_cols[0])
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Heatmap
        if len(num_cols) >= 2:
            with st.expander("🔥 Correlation Heatmap"):
                fig_heat, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig_heat)

        # Treemap
        if len(cat_cols) >= 2 and num_cols:
            with st.expander("🌲 Treemap"):
                treemap_path = st.multiselect("Treemap Levels", cat_cols, default=cat_cols[:2])
                treemap_val = st.selectbox("Treemap Value", num_cols, key="treemapval")
                fig_tree = px.treemap(df, path=treemap_path, values=treemap_val, color=treemap_path[-1])
                st.plotly_chart(fig_tree, use_container_width=True)

        # Pivot Table
        if cat_cols and num_cols:
            with st.expander("📋 Pivot Table"):
                row = st.selectbox("Row", cat_cols, key="pivotrow")
                col = st.selectbox("Column", cat_cols, key="pivotcol")
                val = st.selectbox("Value", num_cols, key="pivotval")
                try:
                    pivot = pd.pivot_table(df, index=row, columns=col, values=val, aggfunc='sum', fill_value=0)
                    st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)
                except:
                    st.warning("❗ Invalid pivot table settings")

    # Raw Data View & Download
    with st.expander("📄 View Full Filtered Data"):
        st.dataframe(df, use_container_width=True)

    st.download_button("⬇️ Download Filtered Dataset", data=df.to_csv(index=False).encode('utf-8'),
                       file_name="filtered_data.csv", mime="text/csv")

else:
    st.info("📌 Upload a CSV or Excel file to begin.")
