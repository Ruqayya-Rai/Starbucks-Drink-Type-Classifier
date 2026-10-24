import os
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Starbucks Drink Type Classifier", page_icon="🧋", layout="centered")

# Load trained model
saved = joblib.load(os.path.join(os.path.dirname(__file__), "drink_type_classifier.pkl"))
model = saved["model"]
features = saved["features"]
examples = saved["examples"]

ICONS = {
    "Coffee & Espresso": "☕",
    "Frappuccino": "🧋",
    "Tea": "🍵",
    "Iced Shaken Drink": "🧊",
    "Smoothie": "🥤",
}
SHORT = {"Tazo® Green Tea Latte": "Green Tea Latte", "Banana Chocolate Smoothie": "Smoothie"}
ABOUT = {
    "Coffee & Espresso": "Brewed coffee, lattes, cappuccinos, mochas and macchiatos.",
    "Frappuccino": "Blended ice drinks — usually high in sugar and calories.",
    "Tea": "Tazo® tea, tea lattes and chai — light to medium calories.",
    "Iced Shaken Drink": "Shaken iced teas and lemonades — light and refreshing.",
    "Smoothie": "Fruit smoothies with milk and protein — high in protein.",
}

# ---------- Styles: Neumorphism (Soft UI) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

:root { --bg: #e4e9f0; --dark: #bcc5d1; --light: #ffffff; --text: #3d4a5c; --muted: #7b8798; --accent: #8a6a52; }
html, body, [class*="css"], .stMarkdown, p, label, input, button { font-family: 'Nunito', sans-serif !important; }
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 3rem; }

.neu { background: var(--bg); border-radius: 30px; box-shadow: 12px 12px 24px var(--dark), -12px -12px 24px var(--light); }
.neu-in { background: var(--bg); border-radius: 22px; box-shadow: inset 7px 7px 14px var(--dark), inset -7px -7px 14px var(--light); }

.hero { text-align: center; padding: 34px 24px 30px; margin-bottom: 30px; }
.orb { width: 108px; height: 108px; border-radius: 50%; margin: 0 auto 18px; display: flex; align-items: center;
  justify-content: center; font-size: 50px; background: var(--bg);
  box-shadow: 10px 10px 20px var(--dark), -10px -10px 20px var(--light), inset 0 0 0 6px var(--bg); }
.hero h1 { color: var(--text); font-weight: 900; font-size: clamp(30px, 6vw, 44px); margin: 0; letter-spacing: -.5px; }
.hero p { color: var(--muted); font-size: 16px; margin: 8px auto 0; max-width: 540px; }
.chips { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; margin-top: 22px; }
.chip { padding: 10px 18px; border-radius: 999px; color: var(--text); font-weight: 800; font-size: 14px;
  box-shadow: inset 4px 4px 8px var(--dark), inset -4px -4px 8px var(--light); }
.chip span { color: var(--accent); }

.label { color: var(--muted); font-weight: 800; letter-spacing: .14em; text-transform: uppercase; font-size: 12px;
  margin: 8px 6px 14px; }

.st-key-inputs { background: var(--bg); border-radius: 30px; padding: 26px 26px 14px;
  box-shadow: 12px 12px 24px var(--dark), -12px -12px 24px var(--light); margin-bottom: 26px; }
[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 800 !important; }
[data-testid="stNumberInputContainer"] { background: var(--bg) !important; border: none !important; border-radius: 16px !important;
  box-shadow: inset 5px 5px 10px var(--dark), inset -5px -5px 10px var(--light) !important; padding: 3px 6px; }
[data-testid="stNumberInputField"] { background: transparent !important; color: var(--text) !important; font-weight: 700; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background: transparent !important;
  color: var(--muted) !important; border: none !important; }

.stButton > button { background: var(--bg) !important; color: var(--text) !important; border: none !important;
  border-radius: 18px !important; font-weight: 800 !important; padding: 12px 10px !important;
  box-shadow: 6px 6px 12px var(--dark), -6px -6px 12px var(--light) !important; transition: all .18s ease; }
.stButton > button:hover { color: var(--accent) !important; }
.stButton > button:active, .stButton > button:focus:active { box-shadow: inset 5px 5px 10px var(--dark), inset -5px -5px 10px var(--light) !important; }
.st-key-predict button { padding: 16px !important; color: var(--accent) !important; border-radius: 22px !important; }
.st-key-predict button p { font-size: 19px !important; font-weight: 900 !important; color: var(--accent) !important; }

.result { padding: 30px 28px; margin-top: 26px; }
.res-top { display: flex; align-items: center; gap: 22px; }
.res-icon { flex: 0 0 96px; height: 96px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 46px; box-shadow: inset 7px 7px 14px var(--dark), inset -7px -7px 14px var(--light); }
.res-type { color: var(--text); font-size: clamp(24px, 5vw, 32px); font-weight: 900; line-height: 1.1; }
.res-conf { color: var(--accent); font-weight: 800; margin-top: 4px; }
.res-about { color: var(--muted); font-size: 14px; margin-top: 6px; }
.bars { margin-top: 24px; }
.bar { display: grid; grid-template-columns: 170px 1fr 52px; align-items: center; gap: 12px; margin: 12px 0;
  color: var(--text); font-weight: 700; font-size: 14px; }
.track { height: 14px; border-radius: 999px; box-shadow: inset 4px 4px 8px var(--dark), inset -4px -4px 8px var(--light);
  overflow: hidden; padding: 2px; }
.fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #c8a68a, #8a6a52); }
.pct { text-align: right; color: var(--muted); }
@media (max-width: 600px) { .bar { grid-template-columns: 110px 1fr 44px; font-size: 12px; } .res-top { flex-direction: column; text-align: center; } }
.actual { margin-top: 18px; padding: 14px 18px; border-radius: 18px; color: var(--text); font-size: 14px;
  box-shadow: inset 4px 4px 8px var(--dark), inset -4px -4px 8px var(--light); }

.footer { text-align: center; margin-top: 34px; padding: 20px; }
.footer .name { color: var(--text); font-weight: 900; font-size: 22px; }
.footer .roll { color: var(--accent); font-weight: 800; letter-spacing: .14em; font-size: 13px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(f"""
<div class="neu hero">
<div class="orb">🧋</div>
<h1>Drink Type Classifier</h1>
<p>Enter the nutrition values of a Starbucks drink and the Decision Tree guesses what kind of drink it is.</p>
<div class="chips">
<div class="chip">Accuracy <span>{saved['accuracy'] * 100:.0f}%</span></div>
<div class="chip">Drink types <span>{len(saved['class_names'])}</span></div>
<div class="chip">Model <span>Decision Tree</span></div>
</div>
</div>
""", unsafe_allow_html=True)

# ---------- Session state ----------
defaults = {"cal": 200.0, "fat": 5.0, "sugar": 30.0, "protein": 8.0, "caf": 75.0}
if "cal" not in st.session_state:
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state.example = None
    st.session_state.show = False


def use_example(i):
    for k, v in zip(defaults, examples[i]["values"]):
        st.session_state[k] = v
    st.session_state.example = i
    st.session_state.show = True


def predict_clicked():
    st.session_state.example = None
    st.session_state.show = True


def values_changed():
    st.session_state.example = None
    st.session_state.show = False


# ---------- Examples ----------
st.markdown('<div class="label">Try a real drink</div>', unsafe_allow_html=True)
cols = st.columns(len(examples))
for i, (col, ex) in enumerate(zip(cols, examples)):
    with col:
        icon = ICONS.get(ex["actual"], "☕")
        label = SHORT.get(ex["label"], ex["label"])
        st.button(f"{icon} {label}", key=f"ex{i}", on_click=use_example, args=(i,), width="stretch")

st.write("")

# ---------- Inputs ----------
with st.container(key="inputs"):
    st.markdown('<div class="label" style="margin-top:0">Nutrition values</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Calories (kcal)", 0.0, 520.0, step=10.0, key="cal", on_change=values_changed)
        st.number_input("Sugars (g)", 0.0, 90.0, step=1.0, key="sugar", on_change=values_changed)
        st.number_input("Caffeine (mg)", 0.0, 410.0, step=5.0, key="caf", on_change=values_changed)
    with c2:
        st.number_input("Total Fat (g)", 0.0, 20.0, step=0.5, key="fat", on_change=values_changed)
        st.number_input("Protein (g)", 0.0, 20.0, step=0.5, key="protein", on_change=values_changed)

st.button("✨  Classify my drink", key="predict", on_click=predict_clicked, width="stretch")

# ---------- Prediction ----------
if st.session_state.show:
    data = pd.DataFrame([[st.session_state.cal, st.session_state.fat, st.session_state.sugar,
                          st.session_state.protein, st.session_state.caf]], columns=features)
    probs = model.predict_proba(data)[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
    best, conf = ranked[0]

    bars = "".join(
        f'<div class="bar"><span>{ICONS[c]} {c}</span><div class="track"><div class="fill" style="width:{max(p * 100, 3):.0f}%"></div></div>'
        f'<span class="pct">{p * 100:.0f}%</span></div>'
        for c, p in ranked
    )
    actual = ""
    if st.session_state.example is not None:
        ex = examples[st.session_state.example]
        mark = "✅" if ex["actual"] == best else "❌"
        actual = f'<div class="actual">{mark} Real drink: <b>{ex["drink"]}</b> — actual type: <b>{ex["actual"]}</b></div>'

    st.markdown(f"""
<div class="neu result">
<div class="res-top">
<div class="res-icon">{ICONS[best]}</div>
<div>
<div class="res-type">{best}</div>
<div class="res-conf">{conf * 100:.0f}% confident</div>
<div class="res-about">{ABOUT[best]}</div>
</div>
</div>
<div class="bars">{bars}</div>
{actual}
</div>
""", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("""
<div class="neu footer">
<div class="name">Ruqayya Shareef</div>
<div class="roll">ROLL NO. 308 · STARBUCKS ML PROJECT</div>
</div>
""", unsafe_allow_html=True)
