"""Power analysis: how many users do we need to detect the effect

uses Cohen's h and a normal approximation, two-sided, equal split

"""


from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

DEFAULT_BASELINE = 0.61
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

_solver = NormalIndPower()


def n_per_arm(mde_pp: float, baseline: float = DEFAULT_BASELINE,
              power: float = DEFAULT_POWER, alpha: float = DEFAULT_ALPHA) -> int:
    """N needed / arm to detect a lift of mde_pp """
    h = proportion_effectsize(baseline + mde_pp / 100, baseline)
    n = _solver.solve_power(effect_size=h, alpha=alpha, power=power,
                            ratio=1.0, alternative="two-sided")
    return int(np.ceil(n))


def power_at(n: int, mde_pp: float, baseline: float = DEFAULT_BASELINE,
             alpha: float = DEFAULT_ALPHA) -> float:
    """achieved power for a true lift of mde_pp at n/ arm."""
    h = proportion_effectsize(baseline + mde_pp / 100, baseline)
    return float(_solver.solve_power(effect_size=h, alpha=alpha, nobs1=n,
                                     ratio=1.0, alternative="two-sided"))


def detectable_mde(n: int, baseline: float = DEFAULT_BASELINE,
                   power: float = DEFAULT_POWER, alpha: float = DEFAULT_ALPHA) -> float:
    """Smallest lift pp detectable with power at n/arm """
    return float(brentq(
        lambda m: power_at(n, m, baseline, alpha) - power, 0.05, 35.0))


def mde_table(mdes=(1.0, 1.5, 2.0, 3.0, 5.0, 8.0),
              baseline: float = DEFAULT_BASELINE,
              power: float = DEFAULT_POWER,
              alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """required sample size for a range of MDE """
    table = pd.DataFrame({"mde_pp": list(mdes)})
    table["n_per_arm"] = table["mde_pp"].map(
        lambda m: n_per_arm(m, baseline, power, alpha))
    table["total_n"] = 2 * table["n_per_arm"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE)
    parser.add_argument("--mde", type=float, default=2.0,
                        help="minimal detectable effect in percentage points")
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    target = n_per_arm(args.mde, args.baseline, args.power, args.alpha)
    print(f"How many users do we need to detect a {args.mde:g}pp lift?")
    print(f"  Baseline retention   : {args.baseline:.0%}")
    print(f"  power / significance : {args.power:.0%} at alpha = {args.alpha}")
    print(f"  Users needed per arm : {target:,}")
    print(f"  Total users needed   : {2 * target:,}")

    print()
    print("required n by effect size:")
    print(mde_table(baseline=args.baseline, power=args.power,
                    alpha=args.alpha).to_string(index=False))

    print()
    print("Present cohort:")

    for n in (250, 2500):
        p = power_at(n, args.mde, args.baseline, args.alpha)
        flag = "  underpowered" if p < args.power else ""
        print(f"  n={n:,}/arm -> power = {p:.0%}{flag}")
        print(f"    smallest effect reliably detectable: "
              f"{detectable_mde(n, args.baseline, args.power, args.alpha):.1f}pp")


if __name__ == "__main__":
    main()
