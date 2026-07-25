"""
stats readouts for experiment

  * two_proportion_readout — headline frequentist result (lift, Wald CI, pooled two-sided z-test)
  * mean_diff_readout      — difference in means with Welch SE; used for CUPED-adjusted (non-binary) outcomes
  * segment_readout        — per-segment lifts, always exploratory
  * bayesian_readout       — Beta-Binomial posterior companion

"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportions_ztest)

DEFAULT_ALPHA = 0.05

COLUMN_ALIASES = {
    "group": "arm", "variant": "arm", "assignment": "arm",
    "segment": "archetype",
    "retained_30d": "retained", "retention": "retained",
}


def normalize_cohort(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_ALIASES)


def two_proportion_readout(success_t: int, n_t: int, success_c: int, n_c: int,
                           alpha: float = DEFAULT_ALPHA) -> dict:
    """Lift in pp, Wald CI, and two-sided pooled z-test p-value."""
    lift = success_t / n_t - success_c / n_c
    _, p = proportions_ztest([success_t, success_c], [n_t, n_c],
                             alternative="two-sided")
    lo, hi = confint_proportions_2indep(success_t, n_t, success_c, n_c,
                                        method="wald", alpha=alpha)
    return {"lift_pp": lift * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "ci_width_pp": (hi - lo) * 100, "p_value": float(p),
            "significant": bool(p < alpha)}


def mean_diff_readout(y_t: pd.Series, y_c: pd.Series,
                      alpha: float = DEFAULT_ALPHA) -> dict:
    y_t, y_c = pd.Series(y_t).astype(float), pd.Series(y_c).astype(float)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = stats.norm.ppf(1 - alpha / 2)
    p = 2 * stats.norm.sf(abs(diff / se))
    return {"lift_pp": diff * 100, "ci_low_pp": (diff - z * se) * 100,
            "ci_high_pp": (diff + z * se) * 100, "ci_width_pp": 2 * z * se * 100,
            "p_value": float(p), "significant": bool(p < alpha)}


def segment_readout(df: pd.DataFrame, segment_col: str = "archetype",
                    arm_col: str = "arm", outcome_col: str = "retained",
                    alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(segment_col):
        c = g.loc[g[arm_col] == "control", outcome_col]
        t = g.loc[g[arm_col] == "treatment", outcome_col]
        if len(c) < 2 or len(t) < 2:
            continue
        r = two_proportion_readout(int(t.sum()), len(t), int(c.sum()), len(c), alpha)
        rows.append({"segment": seg, "n": len(g),
                     "control_rate": c.mean(), "treatment_rate": t.mean(),
                     "lift_pp": r["lift_pp"], "ci_low_pp": r["ci_low_pp"],
                     "ci_high_pp": r["ci_high_pp"], "p_value": r["p_value"]})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def bayesian_readout(trt: pd.Series, ctrl: pd.Series, draws: int = 200_000,
                     seed: int = 7) -> dict:
    """Beta-Binomial posterior with uniform priors on each arm's rate."""
    rng = np.random.default_rng(seed)
    trt, ctrl = pd.Series(trt), pd.Series(ctrl)
    post_c = rng.beta(ctrl.sum() + 1, (1 - ctrl).sum() + 1, draws)
    post_t = rng.beta(trt.sum() + 1, (1 - trt).sum() + 1, draws)
    lift = (post_t - post_c) * 100
    lo, hi = np.percentile(lift, [2.5, 97.5])
    return {"p_treatment_better": float((lift > 0).mean()),
            "posterior_mean_lift_pp": float(lift.mean()),
            "credible_low_pp": float(lo), "credible_high_pp": float(hi),
            "p_lift_above_2pp": float((lift > 2).mean()),
            "samples_pp": lift}


def print_headline(trt: pd.Series, ctrl: pd.Series,
                   alpha: float = DEFAULT_ALPHA) -> dict:
    r = two_proportion_readout(int(trt.sum()), len(trt),
                               int(ctrl.sum()), len(ctrl), alpha)
    print(f"Control retention   : {ctrl.mean():.1%} (n={len(ctrl):,})")
    print(f"Treatment retention : {trt.mean():.1%} (n={len(trt):,})")
    print(f"Observed lift       : {r['lift_pp']:+.1f}pp")
    print(f"95% confidence range: [{r['ci_low_pp']:+.1f}pp, {r['ci_high_pp']:+.1f}pp]")
    print(f"p-value             : {r['p_value']:.4f}")
    print()
    if r["significant"]:
        print(f"Significant at alpha = {alpha}.")
    else:
        print("Not significant.")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="data/cohort.csv")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    cohort = normalize_cohort(pd.read_csv(args.cohort))
    ctrl = cohort.loc[cohort["arm"] == "control", "retained"]
    trt = cohort.loc[cohort["arm"] == "treatment", "retained"]

    print_headline(trt, ctrl, args.alpha)

    bayes = bayesian_readout(trt, ctrl)
    print()
    print("Bayesian readout: Beta Binomial, uniform priors)")
    print(f"  P(treatment > control)  : {bayes['p_treatment_better']:.1%}")
    print(f"  Posterior mean lift     : {bayes['posterior_mean_lift_pp']:+.1f}pp")
    print(f"  95% credible interval   : [{bayes['credible_low_pp']:+.1f}pp, "
          f"{bayes['credible_high_pp']:+.1f}pp]")

    if "archetype" in cohort.columns:
        print()
        print("By segment (exploratory)")
        print(segment_readout(cohort, alpha=args.alpha).round(3)
              .to_string(index=False))


if __name__ == "__main__":
    main()
