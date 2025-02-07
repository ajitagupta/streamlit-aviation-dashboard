import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson, norm

# Load sample aviation incident data (simulated dataset)
data = {
    "Year": np.random.randint(2000, 2024, 100),
    "Incidents": np.random.poisson(lam=10, size=100),
    "Fatalities": np.random.randint(0, 200, 100),
    "Aircraft_Type": np.random.choice(["Boeing 737", "Airbus A320", "Embraer E190"], 100),
    "Airline": np.random.choice(["Airline A", "Airline B", "Airline C"], 100),
    "Weather_Conditions": np.random.choice(["Clear", "Foggy", "Stormy"], 100)
}

df = pd.DataFrame(data)

# Streamlit App
st.title("Aviation Risk Modeling Dashboard")

# Sidebar for user input
st.sidebar.header("User Options")
if st.sidebar.button("Show Data Summary"):
    st.write(df.describe())

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(df["Incidents"], bins=20, kde=True, color="blue", ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Aviation Incidents")
axes[0, 0].set_xlabel("Number of Incidents")
axes[0, 0].set_ylabel("Frequency")

sns.boxplot(x=df["Weather_Conditions"], y=df["Incidents"], palette="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("Incidents by Weather Conditions")
axes[0, 1].set_xlabel("Weather Conditions")
axes[0, 1].set_ylabel("Number of Incidents")

sns.barplot(x=df["Aircraft_Type"].value_counts().index, y=df["Aircraft_Type"].value_counts().values, palette="viridis", ax=axes[1, 0])
axes[1, 0].set_title("Aircraft Type Distribution")
axes[1, 0].set_xlabel("Aircraft Type")
axes[1, 0].set_ylabel("Count")
axes[1, 0].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Risk Scoring Model
st.subheader("Risk Scoring Model")
df["Risk_Score"] = df["Incidents"] * 0.5 + df["Fatalities"] * 0.3
df["Risk_Score"] = df["Risk_Score"].apply(lambda x: round(x, 2))
st.write("Top 10 High-Risk Flights:")
st.write(df.sort_values(by="Risk_Score", ascending=False).head(10))

# Monte Carlo Simulation for Loss Prediction
st.subheader("Monte Carlo Simulation for Loss Prediction")
n_simulations = st.sidebar.slider("Number of Simulations", min_value=500, max_value=5000, step=500, value=1000)
simulated_losses = []
for _ in range(n_simulations):
    simulated_incidents = poisson.rvs(mu=df["Incidents"].mean(), size=len(df))
    simulated_fatalities = norm.rvs(loc=df["Fatalities"].mean(), scale=df["Fatalities"].std(), size=len(df))
    total_loss = np.sum(simulated_incidents * 50000 + simulated_fatalities * 200000)
    simulated_losses.append(total_loss)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(simulated_losses, bins=30, kde=True, color="green", ax=ax)
ax.set_title("Monte Carlo Simulation of Aviation Losses")
ax.set_xlabel("Total Estimated Loss ($)")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.write(f"Estimated Mean Loss: ${np.mean(simulated_losses):,.2f}")
