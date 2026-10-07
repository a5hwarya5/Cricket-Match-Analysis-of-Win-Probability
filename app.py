import pickle

import pandas as pd
import streamlit as st


st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 820px; padding-top: 2rem;}
    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.6rem 1.2rem; border-radius: 16px; text-align: center;
        color: white; margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem; color: white;}
    .hero p {margin: .3rem 0 0; opacity: .85;}
    .team-card {
        border-radius: 14px; padding: 1.1rem; text-align: center;
        background: rgba(128,128,128,.12); border: 1px solid rgba(128,128,128,.25);
    }
    .team-card h4 {margin: 0 0 .4rem; font-size: 1rem;}
    .team-card .pct {font-size: 2.4rem; font-weight: 700; margin: 0;}
    .win {border-color: #2ecc71;}
    .win .pct {color: #2ecc71;}
    .lose .pct {color: #e74c3c;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🏏 IPL Win Predictor</h1>'
    "<p>Live second-innings win probability</p></div>",
    unsafe_allow_html=True,
)

TEAMS = sorted([
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals",
])

CITIES = sorted([
    "Hyderabad", "Bangalore", "Mumbai", "Indore", "Kolkata", "Delhi",
    "Chandigarh", "Jaipur", "Chennai", "Cape Town", "Port Elizabeth",
    "Durban", "Centurion", "East London", "Johannesburg", "Kimberley",
    "Bloemfontein", "Ahmedabad", "Cuttack", "Nagpur", "Dharamsala",
    "Visakhapatnam", "Pune", "Raipur", "Ranchi", "Abu Dhabi",
    "Sharjah", "Mohali", "Bengaluru",
])


@st.cache_resource
def load_model():
    with open("pipe.pkl", "rb") as f:
        return pickle.load(f)


try:
    pipe = load_model()
except FileNotFoundError:
    st.error("`pipe.pkl` not found. Place it in the same folder as this app.")
    st.stop()
    
st.subheader("Match setup")
c1, c2 = st.columns(2)
with c1:
    batting_team = st.selectbox("🏏 Batting team", TEAMS)
with c2:
    bowling_options = [t for t in TEAMS if t != batting_team]
    bowling_team = st.selectbox("🎯 Bowling team", bowling_options)

selected_city = st.selectbox("📍 Host city", CITIES)
target = st.number_input("Target (runs)", min_value=1, max_value=300, value=180, step=1)

st.subheader("Current match situation")
c3, c4, c5, c6 = st.columns(4)
with c3:
    score = st.number_input("Score", min_value=0, max_value=400, value=0, step=1)
with c4:
    overs = st.number_input("Overs done", min_value=0, max_value=20, value=0, step=1)
with c5:
    balls = st.number_input("Balls (in over)", min_value=0, max_value=5, value=0, step=1)
with c6:
    wickets_out = st.number_input("Wickets out", min_value=0, max_value=10, value=0, step=1)

st.write("")
predict = st.button("🔮 Predict Probability", use_container_width=True, type="primary")


if predict:
    balls_bowled = overs * 6 + balls
    balls_left = 120 - balls_bowled
    runs_left = target - score
    wickets_left = 10 - wickets_out


    if balls_bowled > 120:
        st.error("Overs cannot exceed 20.")
        st.stop()
    if balls_bowled == 0:
        st.warning("Enter at least one ball bowled to calculate run rates.")
        st.stop()

    if runs_left <= 0:
        st.success(f"🎉 {batting_team} have already reached the target!")
        st.stop()
    if wickets_left == 0:
        st.error(f"All out! {bowling_team} win.")
        st.stop()
    if balls_left == 0:
        st.error(f"Innings over. {bowling_team} win by {runs_left - 1} run(s).")
        st.stop()

    crr = score * 6 / balls_bowled
    rrr = runs_left * 6 / balls_left

    input_df = pd.DataFrame({
        "batting_team": [batting_team],
        "bowling_team": [bowling_team],
        "city": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets_left],
        "total_runs_x": [target],
        "crr": [crr],
        "rrr": [rrr],
    })

    result = pipe.predict_proba(input_df)
    loss = float(result[0][0])
    win = float(result[0][1])

    # ---- match summary
    st.subheader("Match summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs left", runs_left)
    m2.metric("Balls left", balls_left)
    m3.metric("Current RR", f"{crr:.2f}")
    m4.metric("Required RR", f"{rrr:.2f}", delta=f"{crr - rrr:+.2f}")

    # ---- probabilities
    st.subheader("Win probability")
    r1, r2 = st.columns(2)
    bat_class, bowl_class = ("win", "lose") if win >= loss else ("lose", "win")
    with r1:
        st.markdown(
            f'<div class="team-card {bat_class}"><h4>{batting_team}</h4>'
            f'<p class="pct">{win * 100:.0f}%</p></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="team-card {bowl_class}"><h4>{bowling_team}</h4>'
            f'<p class="pct">{loss * 100:.0f}%</p></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.progress(win, text=f"{batting_team} {win * 100:.0f}%  vs  {loss * 100:.0f}% {bowling_team}")