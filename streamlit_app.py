import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio("Go to", ["📊 Analysis", "🔮 Prediction"])

# Paths to models and dataset
scaler_path = "Model/scaler.pkl"
model_path = "Model/ridge.pkl"

# Load models
@st.cache_resource
def load_models():
    try:
        scaler = pickle.load(open(scaler_path, "rb"))
        model = pickle.load(open(model_path, "rb"))
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.stop()

Standard_scaler, ridge_model = load_models()

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dataset/Algerian_forest_fires_dataset_cleaned.csv")
        df.Classes = np.where(df.Classes.str.contains('not fire'), 0, 1)
        X = df.drop(["FWI", "BUI", "DC", "day", "month", "year"], axis=1).reset_index(drop=True)  # Input data
        y = df["FWI"]  # Target data
        return X, y, df
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        st.stop()

X, y, df = load_data()

# Scale data for visualization
scaled_data = pd.DataFrame(Standard_scaler.transform(X), columns=X.columns)

# Analysis Page
if page == "📊 Analysis":
    st.title("📈 Data Analysis Dashboard")
    st.markdown("Gain insights into the **forest fire dataset** with various visualizations.")

    # Display basic dataset information
    st.subheader("📜 Dataset Overview")
    with st.expander("View Dataset Details"):
        st.dataframe(X.head())
        st.write(f"**Dataset shape:** {X.shape}")
        st.write("**Statistical Summary:**")
        st.write(X.describe())

    # Correlation Plot
    st.subheader("🔗 Correlation Plot")
    fig, ax = plt.subplots()
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Box Plot (Scaled)
    st.subheader("📦 Box Plot (Scaled Data)")
    fig, ax = plt.subplots()
    sns.boxplot(data=scaled_data, palette="Set3", ax=ax)
    st.pyplot(fig)

    # Histogram
    st.subheader("📊 Feature Distribution (Histogram)")
    st.markdown("Visualize the distribution of features across the dataset.")
    fig = X.hist(bins=30, figsize=(20, 15), color="skyblue", edgecolor="black")
    st.pyplot(plt.gcf())

    # Pie Chart
    st.subheader("🔥 Fire vs Non-Fire (Pie Chart)")
    if "Classes" in X.columns:
        percentage = X["Classes"].value_counts() / X["Classes"].sum() * 100
        fig, ax = plt.subplots()
        ax.pie(percentage, labels=["Fire", "Not Fire"], autopct="%1.1f%%", colors=["red", "green"])
        ax.set_title("Percentage of Fire vs Non-Fire")
        st.pyplot(fig)

    # Monthly Fire Analysis
    st.subheader("📅 Monthly Fire Analysis")
    st.markdown("Analyze forest fires across months in **Sidi-Bel** and **Brjaia** regions.")
    regions = {"Sidi-Bel Regions": 1, "Brjaia Regions": 0}

    for region_name, region_value in regions.items():
        st.markdown(f"### {region_name}")
        region_df = df[df["Region"] == region_value]
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.countplot(x="month", hue="Classes", data=region_df, palette="coolwarm", ax=ax)
        ax.set_title(f"Monthly Fire Analysis in {region_name}")
        st.pyplot(fig)

    # Observations
    st.subheader("📌 Observations")
    sidi_bel_counts = df[df.Region == 1].Classes.value_counts()
    brjaia_counts = df[df.Region == 0].Classes.value_counts()

    st.write("- The month of **July** & **August** has the highest number of forest fires, nearly a count of **50** in both regions.")
    st.write("- Both regions had their least no-fire count in the months of **June** & **September**.")
    st.write(f"- The number of fires in **Sidi-Bel Regions** is more than in **Brjaia Regions** by a count of **{sidi_bel_counts[1] - brjaia_counts[1]}** from June to September.")

# Prediction Page
elif page == "🔮 Prediction":
    st.title("🔥 Forest Fire Prediction")
    st.markdown("Provide input values to predict the **Fire Weather Index (FWI)**.")
    
    # Collect user input
    user_input = {}
    for col in X.columns[:-1]:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        default_val = (min_val + max_val) / 2  # Default midpoint
        user_input[col] = st.slider(f"{col}", min_val, max_val, default_val)
    
    Region = st.selectbox("🌍 Select Region", ["Sidi-Bel Regions", "Brjaia Regions"], index=0, key="region_select")
    user_input["Region"] = 1 if Region == "Sidi-Bel Regions" else 0

    input_df = pd.DataFrame([user_input])

    # Scale input and make predictions
    try:
        scaled_input = Standard_scaler.transform(input_df)
        result = ridge_model.predict(scaled_input)
        st.write(f"### 🔥 Predicted Fire Weather Index (FWI): {result[0]:.3f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
