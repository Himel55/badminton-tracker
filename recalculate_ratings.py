#!/usr/bin/env python3
"""Recalculate TrueSkill mu/sigma for players from badminton_data.json matches.

Usage:
  python recalculate_ratings.py [--input badminton_data.json] [--output badminton_data.json]

This script:
  - Loads the JSON file (default: badminton_data.json)
  - Initializes all players with default TrueSkill ratings
  - Replays matches in chronological order and updates ratings
  - Updates each player's `mu`, `sigma`, and `gp` (games played)
  - Writes the updated JSON (backups original to <file>.bak)

Requires: pip install trueskill
"""

import json
import argparse
import shutil
from trueskill import Rating, TrueSkill


def parse_args():
    p = argparse.ArgumentParser(description="Recalculate TrueSkill ratings from match history")
    p.add_argument("--input", "-i", default="badminton_data.json", help="Input JSON file")
    p.add_argument("--output", "-o", default=None, help="Output JSON file (default: overwrite input)")
    p.add_argument("--precision", "-p", type=int, default=12, help="Number of decimal places for mu/sigma")
    p.add_argument("--dry-run", action="store_true", help="Don't write file, just print results")
    return p.parse_args()


def rate_with_score(winning_team_ratings, losing_team_ratings, winner_score, loser_score, ts=None, min_mul=0.8, max_mul=1.2):
    """Rate two teams and adjust mu changes by a multiplier based on point spread.

    Returns (updated_winning_team_ratings, updated_losing_team_ratings)
    """
    if ts is None:
        ts = TrueSkill()

    # Standard updates
    new_win_team, new_loss_team = ts.rate([winning_team_ratings, losing_team_ratings], ranks=[0, 1])

    # Compute multiplier from point differential
    point_diff = float(winner_score) - float(loser_score)
    ratio = point_diff / float(max(winner_score, 1))
    multiplier = min_mul + ratio * (max_mul - min_mul)
    multiplier = max(min_mul, min(max_mul, multiplier))

    final_winner = []
    final_loser = []
    for old, new in zip(winning_team_ratings, new_win_team):
        delta = new.mu - old.mu
        final_winner.append(Rating(mu=old.mu + (delta * multiplier), sigma=new.sigma))
    for old, new in zip(losing_team_ratings, new_loss_team):
        delta = old.mu - new.mu
        final_loser.append(Rating(mu=old.mu - (delta * multiplier), sigma=new.sigma))
    return final_winner, final_loser


def main():
    args = parse_args()
    infile = args.input
    outfile = args.output or infile

    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all players from friends and matches
    players = set()
    if isinstance(data.get("friends"), dict):
        players.update(data["friends"].keys())

    for m in data.get("matches", []):
        players.update(m.get("teamA", []))
        players.update(m.get("teamB", []))

    # Initialize ratings, game counts and win counts
    ratings = {p: Rating() for p in players}
    gp = {p: 0 for p in players}
    wins = {p: 0 for p in players}

    # Replay matches in order
    for match in data.get("matches", []):
        teamA = match.get("teamA", [])
        teamB = match.get("teamB", [])
        ptsA = match.get("ptsA")
        ptsB = match.get("ptsB")

        if ptsA is None or ptsB is None:
            # try to compute from scores if ptsA/ptsB absent
            scores = match.get("scores", [])
            if scores and isinstance(scores[0], list) and len(scores[0]) >= 2:
                ptsA, ptsB = scores[0][0], scores[0][1]
            else:
                print("Skipping match with missing scores:", match)
                continue

        # Build rating lists for teams
        teamA_ratings = [ratings.get(name, Rating()) for name in teamA]
        teamB_ratings = [ratings.get(name, Rating()) for name in teamB]

        # Determine ranks: lower rank is better (0 = winner)
        if ptsA > ptsB:
            ranks = [0, 1]
        elif ptsA < ptsB:
            ranks = [1, 0]
        else:
            ranks = [0, 0]  # draw

        # Rate (with score-based multiplier)
        if ptsA == ptsB:
            new_ratings = TrueSkill().rate([teamA_ratings, teamB_ratings], ranks=[0, 0])
            new_teamA, new_teamB = new_ratings[0], new_ratings[1]
        elif ptsA > ptsB:
            new_teamA, new_teamB = rate_with_score(teamA_ratings, teamB_ratings, ptsA, ptsB)
        else:
            new_teamB, new_teamA = rate_with_score(teamB_ratings, teamA_ratings, ptsB, ptsA)

        for name, newr in zip(teamA, new_teamA):
            ratings[name] = newr
            gp[name] = gp.get(name, 0) + 1
        for name, newr in zip(teamB, new_teamB):
            ratings[name] = newr
            gp[name] = gp.get(name, 0) + 1

        # Increment wins for members of winning team
        if ptsA > ptsB:
            for name in teamA:
                wins[name] = wins.get(name, 0) + 1
        elif ptsB > ptsA:
            for name in teamB:
                wins[name] = wins.get(name, 0) + 1

    # Update friends section
    friends = data.setdefault("friends", {})
    prec = args.precision
    for p in sorted(players):
        r = ratings[p]
        friends[p] = {
            "mu": round(float(r.mu), prec),
            "sigma": round(float(r.sigma), prec),
            "gp": int(gp.get(p, 0)),
            "wins": int(wins.get(p, 0)),
        }

    # Print summary
    print("Recomputed ratings (sorted by mu):")
    summary = sorted(friends.items(), key=lambda kv: kv[1]["mu"], reverse=True)
    for name, info in summary:
        print(f"  {name:12s} mu={info['mu']:.6f} sigma={info['sigma']:.6f} gp={info['gp']} wins={info['wins']}")

    if args.dry_run:
        print("Dry run, not writing file.")
        return

    # Backup and write
    backup = outfile + ".bak"
    shutil.copyfile(infile, backup)
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote updated data to {outfile} (backup saved as {backup})")


if __name__ == "__main__":
    main()
