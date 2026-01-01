import math
import random
import numpy as np
from scipy.stats import pearsonr
import trueskill
import matplotlib.pyplot as plt

MATCHES_COUNT = 5000
PLAYERS_COUNT = 250
UPDATE_EVERY = 20
PAUSE = 0.01

# ==========================================
# 1. YOUR CUSTOM ALGORITHM
# ==========================================
class UserRankingSystem:
    def __init__(self):
        self.players = {} 
        self.params = {
            'INIT_MU': 25.0,
            'INIT_SIGMA': 8.33,
            'BASE_K': 6.0,
            'SIGMA_DECAY': 0.99,
            'MIN_SIGMA': 2.0
        }

    def get_player(self, pid):
        if pid not in self.players:
            self.players[pid] = {
                'mu': self.params['INIT_MU'], 
                'sigma': self.params['INIT_SIGMA'], 
                'gp': 0
            }
        return self.players[pid]

    def predict_win_prob(self, teamA, teamB):
        # Logistic prediction based on team Mu sum
        muA = sum([self.get_player(p)['mu'] for p in teamA])
        muB = sum([self.get_player(p)['mu'] for p in teamB])
        return 1.0 / (1.0 + math.exp(-(muA - muB) / 10.0)) 

    def update(self, teamA, teamB, scores):
        ptsA = scores[0]
        ptsB = scores[1]
        total = ptsA + ptsB
        if total <= 0: return

        # Team Stats
        sumMuA = sum([self.get_player(p)['mu'] for p in teamA])
        sumMuB = sum([self.get_player(p)['mu'] for p in teamB])
        dMu = sumMuA - sumMuB

        # Calculations
        Ep = 1.0 / (1.0 + 10**(-dMu * (16/400.0)))
        Sp = ptsA / total
        
        # MOV Logic
        mov = max(0.05, min(1.0, abs(Sp - 0.5) * 2.0))
        damp_x = dMu * (16/400.0)
        damp = 2.2 / (0.001 * abs(damp_x) + 2.2)
        scale = mov * damp
        deltaBase = (Sp - Ep) * scale

        # Update Team A
        for p in teamA:
            stats = self.get_player(p)
            k = self.params['BASE_K'] * (stats['sigma'] / self.params['INIT_SIGMA'])
            stats['mu'] += k * deltaBase
            stats['sigma'] = max(self.params['MIN_SIGMA'], stats['sigma'] * self.params['SIGMA_DECAY'])
            stats['gp'] += 1
            
        # Update Team B
        for p in teamB:
            stats = self.get_player(p)
            k = self.params['BASE_K'] * (stats['sigma'] / self.params['INIT_SIGMA'])
            stats['mu'] -= k * deltaBase
            stats['sigma'] = max(self.params['MIN_SIGMA'], stats['sigma'] * self.params['SIGMA_DECAY'])
            stats['gp'] += 1

# ==========================================
# 2. MICROSOFT TRUESKILL (The Challenger)
# ==========================================
class TrueSkillSystem:
    def __init__(self):
        # Setup environment (mu=25, sigma=8.333 to match your scale)
        self.env = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.167, tau=0.083, draw_probability=0.0)
        self.players = {}

    def get_rating(self, pid):
        if pid not in self.players:
            self.players[pid] = self.env.create_rating()
        return self.players[pid]

    def predict_win_prob(self, teamA, teamB):
        # TrueSkill logic: Win Prob = P(PerformanceA > PerformanceB)
        # For simplicity in 1v1, we use the delta_mu / denominator formula
        rA = [self.get_rating(p) for p in teamA]
        rB = [self.get_rating(p) for p in teamB]
        delta_mu = sum(r.mu for r in rA) - sum(r.mu for r in rB)
        sum_sigma = sum(r.sigma * 2 for r in rA) + sum(r.sigma * 2 for r in rB)
        size = len(teamA) + len(teamB)
        denom = math.sqrt(size * (self.env.beta * self.env.beta) + sum_sigma)
        return self.env.cdf(delta_mu / denom)

    def update(self, teamA, teamB, scores):
        # TrueSkill library uses ranks (0=winner, 1=loser)
        # It ignores the score margin!
        rA = [self.get_rating(p) for p in teamA]
        rB = [self.get_rating(p) for p in teamB]

        if scores[0] > scores[1]:
            ranks = [0, 1] # Team A wins
        else:
            ranks = [1, 0] # Team B wins
            
        # Update
        updated = self.env.rate([rA, rB], ranks=ranks)
        
        # Save back to player dict
        for i, p in enumerate(teamA): self.players[p] = updated[0][i]
        for i, p in enumerate(teamB): self.players[p] = updated[1][i]

# ==========================================
# 3. ELO (Baseline)
# ==========================================
class EloSystem:
    def __init__(self):
        # Use same rating scale as the other algorithms (mean ~25)
        self.players = {}
        self.INIT = 25.0
        self.K = 6.0

    def get_player(self, pid):
        if pid not in self.players:
            self.players[pid] = {'mu': self.INIT}
        return self.players[pid]

    def get_rating(self, pid):
        return self.get_player(pid)['mu']

    def predict_win_prob(self, teamA, teamB):
        rA = sum(self.get_rating(p) for p in teamA)
        rB = sum(self.get_rating(p) for p in teamB)
        # Standard Elo logistic with 400 scaling
        return 1.0 / (1.0 + 10**((rB - rA) / 4.0))

    def update(self, teamA, teamB, scores):
        ptsA, ptsB = scores
        total = ptsA + ptsB
        if total <= 0: return

        probA = self.predict_win_prob(teamA, teamB)
        sA = 1.0 if ptsA > ptsB else 0.0
        delta = self.K * (sA - probA)

        # Distribute rating change equally among team members
        for p in teamA:
            self.get_player(p)['mu'] += delta / len(teamA)
        for p in teamB:
            self.get_player(p)['mu'] -= delta / len(teamB)

# ==========================================
# 3. SIMULATION RUNNER
# ==========================================
def run_simulation():
    print("--- 1. Generating Synthetic Data ---")
    np.random.seed(69) # For reproducibility
    random.seed(69)
    
    # Players with hidden "True Skill" (Normal dist, mean=25, sd=8)
    true_skills = {i: np.random.normal(25, 8.33) for i in range(PLAYERS_COUNT)}
    pids_sorted = sorted(true_skills.keys())
    xs = np.array([true_skills[pid] for pid in pids_sorted])
    
    matches = []
    print(f"--- 2. Simulating {MATCHES_COUNT} Matches ---")
    for _ in range(MATCHES_COUNT):
        # Pick 4 distinct players for doubles (2v2)
        sampled = random.sample(list(true_skills.keys()), 4)
        teamA = sampled[:2]
        teamB = sampled[2:]

        # Determine Winner based on hidden team skill (sum of members)
        skillA = sum(true_skills[p] for p in teamA)
        skillB = sum(true_skills[p] for p in teamB)
        skill_diff = skillA - skillB
        # Win probability (matches TrueSkill beta assumptions)
        win_prob_A = trueskill.TrueSkill().cdf(skill_diff / math.sqrt(2 * 4.167**2))

        winner_is_A = random.random() < win_prob_A

        # Generate Score (Close skill = Close score)
        # Noise added so the better team doesn't always stomp; scale diff for teams
        loser_score = int(np.clip(np.random.normal(19 - abs(skill_diff)/4, 4), 0, 19))
        scores = [21, loser_score] if winner_is_A else [loser_score, 21]

        matches.append({
            'teamA': teamA, 'teamB': teamB,
            'scores': scores, 'actual_winner': 'A' if winner_is_A else 'B'
        })

    # ==========================================
    # 4. THE TOURNAMENT (run matches sequentially and update plots)
    # ==========================================
    systems = {
        "Your Algo (MOV)": UserRankingSystem(),
        "TrueSkill (Official)": TrueSkillSystem(),
        "Elo (Simple)": EloSystem()
    }

    # Track stats
    correct_predictions = {name: 0 for name in systems}

    # Setup plotting if available
    has_plot = plt is not None
    if has_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {"Your Algo (MOV)": "C0", "TrueSkill (Official)": "C1", "Elo (Simple)": "C2"}

        scatters = {}
        for name, sys in systems.items():
            if name == "TrueSkill (Official)":
                ys = np.array([sys.get_rating(pid).mu for pid in pids_sorted])
            else:
                ys = np.array([sys.get_player(pid)['mu'] for pid in pids_sorted])
            scatters[name] = ax.scatter(xs, ys, label=name, c=colors[name], alpha=0.7)

        ax.plot(xs, xs, 'k--', alpha=0.5)
        ax.set_xlabel('Hidden True Skill')
        ax.set_ylabel('Predicted Skill')
        # Fixed axis scale for clearer convergence visualization
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 60)
        ax.grid(True, linestyle='--', alpha=0.4)
        legend = ax.legend()
        title = ax.set_title('Match 0')
        # Add a text box to display match info (players, hidden skills, score, winner)
        match_info = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=9,
                             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        # Draw and keep the initial state visible for 5 seconds with a countdown
        plt.draw()
        for t in range(5, 0, -1):
            title.set_text(f"Starting in {t}s")
            match_info.set_text("Waiting to start...")
            plt.pause(1.0)
        title.set_text('Match 0')
        match_info.set_text("")

    total_matches = len(matches)

    for i, m in enumerate(matches, start=1):
        for name, sys in systems.items():
            probA = sys.predict_win_prob(m['teamA'], m['teamB'])
            predicted = 'A' if probA > 0.5 else 'B'
            if predicted == m['actual_winner']:
                correct_predictions[name] += 1
            sys.update(m['teamA'], m['teamB'], m['scores'])

        # Update visuals periodically
        if has_plot and (i % UPDATE_EVERY == 0 or i == total_matches):
            for name, sys in systems.items():
                if name == "TrueSkill (Official)":
                    ys = np.array([sys.get_rating(pid).mu for pid in pids_sorted])
                else:
                    ys = np.array([sys.get_player(pid)['mu'] for pid in pids_sorted])
                offsets = np.column_stack((xs, ys))
                scatters[name].set_offsets(offsets)
                # Sanity check after updating offsets
                offs = scatters[name].get_offsets()
                if offs.shape[0] != len(pids_sorted):
                    raise RuntimeError(f"Plot update sanity check failed: '{name}' has {offs.shape[0]} points; expected {len(pids_sorted)}")
                if not np.allclose(offs[:, 0], xs):
                    print(f"Warning: x coordinates for '{name}' changed during update (may be shifted or jittered)")

            # Update legend labels with current corr and accuracy
            labels = []
            for name, sys in systems.items():
                if name == "TrueSkill (Official)":
                    ests = np.array([sys.get_rating(pid).mu for pid in pids_sorted])
                else:
                    ests = np.array([sys.get_player(pid)['mu'] for pid in pids_sorted])
                corr, _ = pearsonr(ests, xs)
                acc = correct_predictions[name] / i
                rmse = float(np.sqrt(np.mean((ests - xs) ** 2)))
                labels.append(f"{name} (corr={corr:.3f}, acc={acc:.2%}, rmse={rmse:.2f})")

            handles = [scatters[name] for name in systems]
            ax.legend(handles, labels)
            # Update match info box with the last processed match
            a_ids = ','.join(str(pid) for pid in m['teamA'])
            b_ids = ','.join(str(pid) for pid in m['teamB'])
            a_skill = sum(true_skills[p] for p in m['teamA'])/len(m['teamA'])
            b_skill = sum(true_skills[p] for p in m['teamB'])/len(m['teamB'])
            winner = m['actual_winner']
            score = f"{m['scores'][0]}-{m['scores'][1]}"
            match_info.set_text(f"Match {i}: A [{a_ids}] ({a_skill:.1f}) vs B [{b_ids}] ({b_skill:.1f})\nScore: {score} Winner: {winner}")
            title.set_text(f"Match {i}/{total_matches}")
            plt.pause(PAUSE)

    # Final results
    print(f"{'System':<20} | {'Accuracy':<10} | {'Correlation':<12} | {'RMSE':<6}")
    print("-" * 65)
    for name, sys in systems.items():
        if name == "TrueSkill (Official)":
            estimated = np.array([sys.get_rating(pid).mu for pid in pids_sorted])
        else:
            estimated = np.array([sys.get_player(pid)['mu'] for pid in pids_sorted])
        truth = np.array([true_skills[pid] for pid in pids_sorted])
        corr, _ = pearsonr(estimated, truth)
        rmse = float(np.sqrt(np.mean((estimated - truth) ** 2)))
        print(f"{name:<20} | {correct_predictions[name]/len(matches):.2%}     | {corr:.3f}       | {rmse:.2f}")

    # Keep plot open until user closes the window
    if has_plot:
        plt.ioff()
        print("Simulation complete — close the plot window to exit.")
        plt.show(block=True)

if __name__ == "__main__":
    run_simulation()