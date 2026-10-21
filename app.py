import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

NUTRIENTS = ["calories", "protein", "carbs", "fat", "fiber"]

# Step 2 + 3: Dataset load & data processing
@st.cache_data
def load_data():
    df = pd.read_csv("foods.csv")
    df = df.dropna().drop_duplicates(subset="food").reset_index(drop=True)
    return df

# Step 4: ML analysis (nearest neighbours on nutrition values)
def find_alternatives(df, food_name, top_n=3):
    scaled = StandardScaler().fit_transform(df[NUTRIENTS])
    model = NearestNeighbors(n_neighbors=len(df)).fit(scaled)
    idx = df.index[df["food"] == food_name][0]
    _, neighbours = model.kneighbors([scaled[idx]])
    selected = df.loc[idx]

    # Step 6: only cheaper foods
    result = df.loc[neighbours[0][1:]]
    result = result[result["price_per_100g"] < selected["price_per_100g"]].head(top_n)
    result = result.assign(
        saving=selected["price_per_100g"] - result["price_per_100g"]
    )
    return selected, result

# Step 1: User food input
st.set_page_config(page_title="AI Nutrition Assistant", page_icon="🍎")
st.title("🍎 AI Nutrition Assistant")
st.caption("Nutrition info + budget-friendly alternatives")

df = load_data()
food = st.selectbox("Select a food", sorted(df["food"]))

if st.button("Analyze"):
    selected, alts = find_alternatives(df, food)

    # Step 5: Nutrition result
    st.subheader(f"Nutrition of {food} (per 100 g)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Calories", f"{selected['calories']:.0f} kcal")
    c2.metric("Protein", f"{selected['protein']} g")
    c3.metric("Carbs", f"{selected['carbs']} g")
    c4, c5, c6 = st.columns(3)
    c4.metric("Fat", f"{selected['fat']} g")
    c5.metric("Fiber", f"{selected['fiber']} g")
    c6.metric("Price", f"₹{selected['price_per_100g']:.0f}")

    # Step 7: Website output
    st.subheader("💰 Budget-friendly alternatives")
    if alts.empty:
        st.info("This food is already budget-friendly! No cheaper similar food found.")
    else:
        for _, r in alts.iterrows():
            st.success(
                f"**{r['food']}** – ₹{r['price_per_100g']:.0f}/100g "
                f"(save ₹{r['saving']:.0f}) | "
                f"{r['calories']:.0f} kcal, {r['protein']}g protein"
            )
        st.bar_chart(
            pd.concat([selected.to_frame().T, alts])
            .set_index("food")[["protein", "carbs", "fat", "fiber"]].astype(float)
        )
