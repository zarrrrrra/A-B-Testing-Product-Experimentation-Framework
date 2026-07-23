"""Fetch developer profiles from the GitHub public API.
For each sampled developer we collect:
    developer_id      
    github_login      
    account_age_days  
    repo_count        
    top_language      
    commit_streak

Use: python -m data.github_fetch --n 500 --out data/profiles.csv
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from datetime import datetime, timezone

import pandas as pd
import requests

API = "https://api.github.com"
# GitHub user ids run well past 150M; sampling below that keeps hit rates high.
MAX_USER_ID_HINT = 150_000_000


def _session(token: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    s.headers["User-Agent"] = "experimentation-framework-demo"
    token = token or os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _get(session: requests.Session, url: str, **params) -> requests.Response:
    """GET with polite rate-limit handling."""
    while True:
        resp = session.get(url, params=params or None, timeout=30)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 1)
            if wait > 300:
                raise RuntimeError(
                    f"GitHub rate limit exhausted; resets in {wait / 60:.0f} min. "
                    "Set GITHUB_TOKEN for 5,000 requests/hour."
                )
            print(f"  rate limited — sleeping {wait:.0f}s "
                  "(set GITHUB_TOKEN to avoid this)")
            time.sleep(wait)
            continue
        return resp


def _commit_streak(session: requests.Session, login: str) -> int:
    """Consecutive days with push activity, ending today (UTC).

    Built from the public events feed, which only reaches back ~90 days /
    300 events — so the streak is effectively capped there. That cap is
    fine for our purpose: 'is this person actively committing right now'.
    """
    resp = _get(session, f"{API}/users/{login}/events/public", per_page=100)
    if resp.status_code != 200:
        return 0
    push_days = {
        ev["created_at"][:10]
        for ev in resp.json()
        if ev.get("type") == "PushEvent" and ev.get("created_at")
    }
    if not push_days:
        return 0
    today = datetime.now(timezone.utc).date()
    streak = 0
    day = today
    while day.isoformat() in push_days:
        streak += 1
        day = day.fromordinal(day.toordinal() - 1)
    return streak


def _top_language(session: requests.Session, login: str) -> str | None:
    resp = _get(session, f"{API}/users/{login}/repos",
                per_page=100, sort="pushed")
    if resp.status_code != 200:
        return None
    langs = Counter(r["language"] for r in resp.json() if r.get("language"))
    return langs.most_common(1)[0][0] if langs else None


def fetch_developer_profiles(n: int = 500, token: str | None = None,
                             seed: int | None = None,
                             verbose: bool = True) -> pd.DataFrame:
    """Sample `n` real developer profiles from the GitHub public API."""
    import numpy as np

    rng = np.random.default_rng(seed)
    session = _session(token)
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    seen: set[int] = set()

    while len(rows) < n:
        since = int(rng.integers(0, MAX_USER_ID_HINT))
        listing = _get(session, f"{API}/users", since=since, per_page=30)
        if listing.status_code != 200:
            continue

        for stub in listing.json():
            if len(rows) >= n:
                break
            if stub.get("type") != "User" or stub["id"] in seen:
                continue
            seen.add(stub["id"])

            detail = _get(session, f"{API}/users/{stub['login']}")
            if detail.status_code != 200:
                continue
            user = detail.json()
            created = datetime.fromisoformat(
                user["created_at"].replace("Z", "+00:00"))

            rows.append({
                "developer_id": f"gh_{user['id']}",
                "github_login": user["login"],
                "account_age_days": (now - created).days,
                "repo_count": user.get("public_repos", 0) or 0,
                "top_language": _top_language(session, user["login"]),
                "commit_streak": _commit_streak(session, user["login"]),
            })
            if verbose and len(rows) % 25 == 0:
                print(f"  fetched {len(rows)}/{n} profiles...")

    profiles = pd.DataFrame(rows)
    if verbose:
        print_profile_spread(profiles)
    return profiles


def print_profile_spread(profiles: pd.DataFrame) -> None:
    yrs = profiles["account_age_days"] / 365.25
    print("Profile spread:")
    print(f"  Median repos        : {profiles['repo_count'].median():.0f}")
    print(f"  Median commit_streak: {profiles['commit_streak'].median():.0f}")
    print(f"  Account age (yrs)   : {yrs.min():.1f} - {yrs.max():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default="data_N500/data/profiles.csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    profiles = fetch_developer_profiles(n=args.n, seed=args.seed)
    profiles.to_csv(args.out, index=False)
    print(f"\nWrote {len(profiles):,} profiles to {args.out}")


if __name__ == "__main__":
    main()
