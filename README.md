# Santa 2020 — Kaggle Competition

## Overview

The [Santa 2020](https://www.kaggle.com/competitions/santa-2020) competition was a multi-armed bandit (MAB) simulation game where two agents competed head-to-head across 100 slot machines over 2,000 rounds. Each machine had a hidden payout probability that decayed with total usage by both players. The challenge was to build an agent that maximized cumulative reward while adapting to opponent behavior and machine decay dynamics — a classic explore-vs-exploit problem with an adversarial twist.

**Result: Peak Rating ~1,300+ (Elo-based leaderboard)**

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Competitive Intelligence

Scraped Meta Kaggle data and leaderboard statistics to analyze the top 100 teams. Built win-rate heatmaps across the top 25 teams, rating-vs-submission-date trajectories, and identified the highest-rated individual agent submissions. This informed which strategies were dominant and which teams to study.

### 2. Heuristic Baseline (Random Sticky Agent)

Built a hand-crafted agent combining exploration with exploitation. The agent tracked per-machine pull counts and success rates, maintained blacklists (machines with too many blanks) and yellowlists (low-performing machines), and used a "stickiness" mechanic — once a machine paid out, the agent stayed on it for several consecutive pulls. Strategy shifted from exploration-heavy in early rounds to pure exploitation after round 1,500.

### 3. Episode Data Pipeline

Built a two-stage data pipeline to learn from top competitors. First, an automated scraper collected 1,200+ episode replays from the top 15 leaderboard teams via the Kaggle Episode API. Then, parser scripts converted raw episode JSON into structured tabular data: per-step features (round number, pull counts, success counts, opponent pulls) paired with payout outcomes.

### 4. ML Agent v1 (Decision Tree + XGBoost)

Trained the first ML-based agent on the scraped episode data. A Decision Tree regressor predicted expected payout per machine given game state features. At each step, the agent selected the machine with maximum predicted payout (with a fudge factor for tie-breaking). Ran champion-vs-challenger simulations to compare Decision Tree against XGBoost variants locally before submitting.

### 5. ML Agent v2 (Phase-Specific Models)

The key insight: optimal strategy shifts dramatically across the 2,000-round game. Trained four separate Decision Tree models for different game phases (rounds 0-500, 500-1000, 1000-1500, 1500-2000). Early phases favor exploration while late phases favor exploitation of learned payout distributions. The agent dynamically swapped models at phase boundaries.

### 6. Final Agent (Top Agent Data + Refined Pipeline)

Scaled training data by scraping episodes from the highest-rated individual agents (beyond just top 15 LB teams). Experimented with feature engineering (game progress ratio, total pulls, success ratios), hyperparameter tuning across multiple model families, and champion-vs-challenger evaluation. The final submission agent used a greedy strategy backed by a trained regressor, updating predictions in real-time as both players' actions were observed.

## Repository Structure

```
├── notebooks/
│   ├── 01-competitive-analysis.ipynb     # Top-100 win-rate heatmaps + rating distributions
│   ├── 02-heuristic-agent.ipynb          # Random-sticky agent with blacklist/yellowlist
│   ├── 03-episode-scraper.ipynb          # Scrapes 1,200+ replays from top 15 LB teams
│   ├── 04-episode-parser.py              # Episode JSON -> tabular training data
│   ├── 05-decision-tree-agent.ipynb      # Decision Tree + XGBoost agent with simulations
│   ├── 06-phase-split-training.ipynb     # 4-phase game-specific model training
│   └── 07-final-model.ipynb             # Final agent: top-agent data + refined pipeline
└── README.md
```

## Tech Stack

- **Models**: scikit-learn (DecisionTreeRegressor), XGBoost, LightGBM, Random Forest
- **Data Collection**: Kaggle Episode API, Meta Kaggle datasets, requests
- **Analysis**: pandas, NumPy, Matplotlib (heatmaps, rating trajectories, reward curves)
- **Simulation**: kaggle-environments (local agent-vs-agent testing)
- **Infrastructure**: Kaggle Notebooks (CPU)

## Competition

- **Name**: [Santa 2020 — The Candy Cane Contest](https://www.kaggle.com/competitions/santa-2020)
- **Type**: Multi-agent simulation (Multi-Armed Bandit)
- **Metric**: Elo-based rating from head-to-head matches
- **Timeline**: December 2020 — January 2021
