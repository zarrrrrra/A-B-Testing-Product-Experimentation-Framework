"""
A/B Testing & Product Experimentation Framework

Interactive dashboard for the autoscaling tip experiment

Run:
    streamlit run app/dashboard.py

Data: expects a cohort CSV (default: data/cohort.csv, the output of data/simulate_cohort.py) 
with the columns 
    arm ('control' / 'treatment')  and  retained (0/1)
    pre_experiment_score (CUPED tab)
    archetype (Segments tab)
    account_age_days, repo_count, commit_streak (cohort tab)

all stats come from the notebooks:
    src/power_analysis.py, src/experiment.py, src/stats.py, src/cuped.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.simulate_cohort import simulate_outcomes 
from src.cuped import cuped_readout 
from src.experiment import srm_check 
from src.power_analysis import detectable_mde, n_per_arm, power_at 
from src.stats import (bayesian_readout, normalize_cohort, 
                       segment_readout, two_proportion_readout)

DEFAULT_COHORT = REPO_ROOT / "data" / "cohort.csv"
ACCENT = "#4C72B0"
ALERT = "#C44E52"

st.set_page_config(
    page_title="A/B Testing and Product Experimentation Framework- Autoscaling Tip Analysis",
    layout="wide",
)
# nav sits at the top of the sidebar, above the data and analysis controls, even though it's populated further down
NAV_SLOT = st.sidebar.container()

# data layer
@st.cache_data(show_spinner=False)
def demo_cohort(n: int = 5000, true_effect_pp: float = 2.0,
                seed: int = 42) -> pd.DataFrame:
    """synthetic profiles run through the real simulator.
    """
    rng = np.random.default_rng(seed)
    profiles = pd.DataFrame({
        "developer_id": [f"demo_{i:05d}" for i in range(n)],
        "account_age_days": rng.gamma(1.3, 2400, n).clip(30, 6700).astype(int),
        "repo_count": rng.negative_binomial(1.2, 0.05, n).clip(0, 400),
        "commit_streak": np.where(rng.random(n) < 0.55, 0,
                                  rng.gamma(1.5, 10, n)).astype(int),
    })
    return simulate_outcomes(profiles, true_effect_pp=true_effect_pp, seed=seed)


def load_cohort() -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader(
            "Cohort CSV", type="csv",
            help="Output of data/simulate_cohort.py or the user_metrics mart.")
    if uploaded is not None:
        return normalize_cohort(pd.read_csv(uploaded)), f"uploaded file · {uploaded.name}"
    if DEFAULT_COHORT.exists():
        return normalize_cohort(pd.read_csv(DEFAULT_COHORT)), "data/cohort.csv"
    return demo_cohort(), "built-in demo data (synthetic)"


# UI: load

cohort, source_label = load_cohort()

missing = {"arm", "retained"} - set(cohort.columns)
if missing:
    st.error(
        f"The cohort file is missing required columns"
    )
    st.stop()

with st.sidebar:
    st.caption(f"Source: {source_label}")
    if "demo" in source_label:
        st.warning("showing demo data- run the pipeline to see real results.")
    st.header("Analysis settings")
    alpha = st.select_slider("Significance level (α)",
                             options=[0.01, 0.05, 0.10], value=0.05)
    st.caption("All statistics come from src/ — the notebooks are the "
               "reference walkthrough of the same functions.")

ctrl = cohort.loc[cohort["arm"] == "control", "retained"]
trt = cohort.loc[cohort["arm"] == "treatment", "retained"]
headline = two_proportion_readout(int(trt.sum()), len(trt),
                                  int(ctrl.sum()), len(ctrl), alpha)
baseline_rate = float(ctrl.mean())
n_smaller_arm = min(len(ctrl), len(trt))

# UI: header

st.title("A/B Testing and Product Experimentation Framework- Autoscaling Tip Analysis")
st.caption(
    "Does surfacing an autoscaling recommendation during onboarding improve "
    "30-day service retention? One row per developer account."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Control retention", f"{ctrl.mean():.1%}", help=f"n = {len(ctrl):,}")
k2.metric("Treatment retention", f"{trt.mean():.1%}", help=f"n = {len(trt):,}")
k3.metric("Observed lift", f"{headline['lift_pp']:+.1f}pp",
          help=f"95% CI [{headline['ci_low_pp']:+.1f}, {headline['ci_high_pp']:+.1f}]pp")
k4.metric("p-value", f"{headline['p_value']:.4f}")

if headline["significant"]:
    st.success(f"Significant at α = {alpha}. The lift is distinguishable from zero.")
else:
    st.info(
        f"**Not significant at α = {alpha}.** The confidence interval "
        f"[{headline['ci_low_pp']:+.1f}pp, {headline['ci_high_pp']:+.1f}pp] includes zero — "
        "this data is consistent with no effect. It is *not* proof of no effect."
    )

# Power reality check
achieved = power_at(n_smaller_arm, 2.0, baseline_rate, alpha)
if achieved < 0.8:
    st.error(
        f"**Power reality check** — at {n_smaller_arm:,} users per arm this "
        f"experiment has **{achieved:.0%} power** to detect the +2pp effect it was designed "
        f"around (target: 80%, requiring "
        f"{n_per_arm(2.0, baseline_rate, alpha=alpha):,} per arm). "
        f"The smallest effect reliably detectable at this size is "
        f"**{detectable_mde(n_smaller_arm, baseline_rate, alpha=alpha):.1f}pp**. "
        "A non-significant readout here was the expected outcome even if the feature works."
    )

SECTIONS = {
    "Headline": "The result, and whether to trust it",
    "Power planner": "How big the next run needs to be",
    "CUPED": "Variance reduction from pre-experiment data",
    "Segments": "Exploratory cuts by developer archetype",
    "Cohort": "Who is actually in this experiment",
}

NAV_SLOT.markdown("""
<style>
/* sidebar nav: radio group > section cards */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}
/* hide radio dot */
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    padding: 0.55rem 0.7rem;
    border: 1px solid rgba(76, 114, 176, 0.22);
    border-left: 4px solid transparent;
    border-radius: 8px;
    background: rgba(76, 114, 176, 0.05);
    font-weight: 600;
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(76, 114, 176, 0.13);
    border-color: rgba(76, 114, 176, 0.45);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(76, 114, 176, 0.20);
    border-color: rgba(76, 114, 176, 0.45);
    border-left-color: #4C72B0;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:focus-within {
    outline: 2px solid #4C72B0;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
    section[data-testid="stSidebar"] div[role="radiogroup"] > label { transition: none; }
}
</style>
""", unsafe_allow_html=True)

with NAV_SLOT:
    st.header("Sections")
    section = st.radio("Sections", list(SECTIONS),
                       label_visibility="collapsed")
    st.caption(SECTIONS[section])
    st.divider()

# section: headline

if section == "Headline":
    left, right = st.columns([3, 2])

    with left:
        ci_df = pd.DataFrame({
            "estimate": ["Observed lift"],
            "lift": [headline["lift_pp"]],
            "lo": [headline["ci_low_pp"]],
            "hi": [headline["ci_high_pp"]],
        })
        band = alt.Chart(ci_df).mark_rule(color=ACCENT, strokeWidth=6, opacity=0.65).encode(
            x=alt.X("lo:Q", title="lift (pp)"), x2="hi:Q", y=alt.Y("estimate:N", title=None))
        point = alt.Chart(ci_df).mark_point(color="#1f3b63", size=110, filled=True).encode(
            x="lift:Q", y="estimate:N")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color=ALERT, strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((band + point + zero).properties(height=110))
        st.caption("Whisker = 95% confidence interval. Crossing the dashed line means "
                   "'no effect' remains consistent with the data.")

    with right:
        st.subheader("Assignment health")
        n_c, n_t = len(ctrl), len(trt)
        srm_p = srm_check(cohort)
        st.write(f"Control {n_c:,} · Treatment {n_t:,} "
                 f"(treatment share {n_t / (n_c + n_t):.1%})")
        if srm_p > 0.001:
            st.write(f"Sample-ratio check: p = {srm_p:.3f} — consistent with 50/50.")
        else:
            st.error(f"Sample-ratio mismatch (p = {srm_p:.2e}). "
                     "Do not read results — find the assignment bug first.")

        bayes = bayesian_readout(trt, ctrl, draws=100_000)
        st.subheader("Bayesian companion")
        st.write(f"P(treatment > control): **{bayes['p_treatment_better']:.1%}**")
        st.write(f"95% credible interval: [{bayes['credible_low_pp']:+.1f}pp, "
                 f"{bayes['credible_high_pp']:+.1f}pp]")
        st.caption("Same data, posterior framing — useful when 'not significant' "
                   "gets misread as 'proven zero'.")

# section: power planner

elif section == "Power planner":
    st.subheader("How big does the next run need to be?")
    c1, c2, c3 = st.columns(3)
    base_in = c1.slider("Baseline retention", 0.30, 0.90, round(baseline_rate, 2), 0.01)
    mde_in = c2.slider("Minimal detectable effect (pp)", 0.5, 10.0, 2.0, 0.5)
    power_in = c3.slider("Target power", 0.70, 0.95, 0.80, 0.05)

    need = n_per_arm(mde_in, base_in, power_in, alpha)
    m1, m2, m3 = st.columns(3)
    m1.metric("Users per arm", f"{need:,}")
    m2.metric("Total users", f"{2 * need:,}")
    m3.metric("Power at current cohort size",
              f"{power_at(n_smaller_arm, mde_in, base_in, alpha):.0%}",
              help="Power the current data would have against this MDE.")

    mdes = np.arange(0.5, 10.01, 0.25)
    curve = pd.DataFrame({
        "MDE (pp)": mdes,
        "users per arm": [n_per_arm(m, base_in, power_in, alpha) for m in mdes],
    })
    chart = alt.Chart(curve).mark_line(color=ACCENT, strokeWidth=2.5).encode(
        x="MDE (pp):Q",
        y=alt.Y("users per arm:Q", scale=alt.Scale(type="log"),
                title="users per arm (log)"),
        tooltip=["MDE (pp)", alt.Tooltip("users per arm:Q", format=",")],
    ).properties(height=320)
    rule = alt.Chart(pd.DataFrame({"x": [mde_in]})).mark_rule(
        color=ALERT, strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart(chart + rule)
    st.caption("Halving the MDE roughly quadruples the required sample. "
               "Choose the effect worth detecting first; the sample size follows.")

#section: CUPED

elif section == "CUPED":
    st.subheader("Variance reduction with pre-experiment data")
    if "pre_experiment_score" not in cohort.columns:
        st.info(
            "This cohort has no `pre_experiment_score` column, so there's nothing to adjust with"
        )
    else:
        cuped = cuped_readout(cohort, outcome_col="retained",
                              covariate_col="pre_experiment_score", alpha=alpha)

        c1, c2, c3 = st.columns(3)
        c1.metric("Correlation (score vs outcome)", f"{cuped['correlation']:.3f}")
        c2.metric("Variance removed", f"{cuped['variance_removed']:.1%}",
                  help="≈ correlation²")
        c3.metric("Theta", f"{cuped['theta']:.3f}")

        table = pd.DataFrame([cuped["naive"], cuped["adjusted"]],
                             index=["Naive (unadjusted)", "CUPED-adjusted"])
        st.dataframe(table.round(4))

        st.write(f"Confidence interval narrowed by **{cuped['ci_narrowing']:.1%}** — "
                 "same effect, measured more precisely.")
        st.caption(
            "CUPED's payoff is the covariate's predictive power: variance removed ≈ r². "
            "A score with r = 0.5 removes 25% of variance — the cheapest power upgrade "
            "available before the next run."
        )

# section: segments

elif section == "Segments":
    st.subheader("Segment breakdown")
    st.warning(
        "Exploratory only. Multiple segments mean multiple chances for a fake "
        "'significant' result, a subgroup finding here could stem from a pre-registered "
        "hypothesis for the next experiment."
    )
    if "archetype" not in cohort.columns:
        st.info("This cohort has no archetype column, no segments to show")
    else:
        seg = segment_readout(cohort, alpha=alpha)

        base = alt.Chart(seg).encode(
            y=alt.Y("segment:N", sort="-x", title=None),
            tooltip=["segment", "n",
                     alt.Tooltip("lift_pp:Q", format="+.1f"),
                     alt.Tooltip("p_value:Q", format=".3f")],
        )
        bars = base.mark_rule(color=ACCENT, strokeWidth=6, opacity=0.65).encode(
            x=alt.X("ci_low_pp:Q", title="lift (pp), 95% CI"), x2="ci_high_pp:Q")
        pts = base.mark_point(color="#1f3b63", size=90, filled=True).encode(x="lift_pp:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color=ALERT, strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((bars + pts + zero).properties(height=60 * len(seg) + 40))

        st.dataframe(
            seg[["segment", "n", "control_rate", "treatment_rate",
                 "lift_pp", "ci_low_pp", "ci_high_pp", "p_value"]].round(3), hide_index=True,
        )
        st.caption("Wide whiskers are the story: small segments cannot support claims, "
                   "whatever their point estimates say.")

#  section: cohort

elif section == "Cohort":
    st.subheader("Who is in this cohort?")
    profile_cols = [c for c in ["account_age_days", "repo_count", "commit_streak",
                                "pre_experiment_score"] if c in cohort.columns]
    if profile_cols:
        pick = st.selectbox("Distribution", profile_cols)
        clip_hi = float(cohort[pick].quantile(0.99))
        hist = alt.Chart(
            cohort.assign(_v=cohort[pick].clip(upper=clip_hi))
        ).mark_bar(color=ACCENT).encode(
            x=alt.X("_v:Q", bin=alt.Bin(maxbins=45), title=f"{pick} (clipped at p99)"),
            y=alt.Y("count()", title="developers"),
        ).properties(height=280)
        st.altair_chart(hist)

    if "archetype" in cohort.columns:
        st.write("Archetype mix:")
        st.dataframe(cohort["archetype"].value_counts().rename("developers"))

    with st.expander("Raw cohort data"):
        st.dataframe(cohort.head(500), hide_index=True)
        st.download_button("Download full cohort as CSV",
                           cohort.to_csv(index=False).encode(),
                           file_name="cohort_export.csv", mime="text/csv")
