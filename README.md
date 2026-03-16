# Santa 2020 — Multi-Armed Bandit Optimization

Building agents that play a competitive multi-armed bandit game against other teams' agents. This was a [Kaggle simulation competition](https://www.kaggle.com/competitions/santa-2020) from the annual Santa series in 2020. Notebooks developed on [Kaggle](https://www.kaggle.com/illidan7).

## Approach

### 1. Competitive Analysis

Scraped and analyzed the top 100 teams' win rates and head-to-head matchup matrices. Identified dominant strategies and exploitable patterns among leading agents, informing the design of counter-strategies.

### 2. Heuristic Agent

Built a baseline agent with a random-sticky strategy (commit to a bandit once it pays out) combined with adaptive blacklisting (abandon bandits with sustained low payouts). Established a performance floor and validated the competition environment.

### 3. Episode Scraper

Automated scraper to download 1000+ game replays from the top 15 teams via the Kaggle API. Collected the raw episode data needed to train supervised models on expert gameplay decisions.

### 4. Episode Parser

Python script to parse raw episode JSON into structured training data: per-step features (bandit payouts, action history, game phase) with labels (chosen action by top agents). Converts replay logs into tabular format suitable for tree-based models.

### 5. Decision Tree Agent

Trained DecisionTree and XGBoost classifiers to predict which bandit a top agent would select given the current game state. The agent selects the bandit with maximum predicted payout, with controlled epsilon-greedy exploration.

### 6. Phase-Split Training

Splits the 2000-step game into 4 phases (exploration, early exploitation, mid-game, endgame) and trains separate models for each. Each phase has distinct optimal strategies — early phases favor exploration while late phases favor exploitation of learned payout distributions.

### 7. Final Model

Production agent combining phase-split XGBoost predictions with fallback heuristics. Selects maximum predicted payout per phase, with adaptive exploration rates that decay as the game progresses.

## Repository Structure

```
santa-2020/
├── README.md
├── .gitignore
└── notebooks/
    ├── 01-competitive-analysis.ipynb               # Top-100 win rates and matchup analysis
    ├── 02-heuristic-agent.ipynb                    # Random-sticky + adaptive blacklisting
    ├── 03-episode-scraper.ipynb                    # Automated replay downloader (top 15 teams)
    ├── 04-episode-parser.py                        # Episode JSON → tabular training data
    ├── 05-decision-tree-agent.ipynb                # DecisionTree/XGBoost bandit predictor
    ├── 06-phase-split-training.ipynb               # 4-phase separate model training
    └── 07-final-model.ipynb                        # Production agent with phase-split XGBoost
```

## Tech Stack

- **ML**: XGBoost, scikit-learn (DecisionTree)
- **Data**: pandas, NumPy
- **Scraping**: Kaggle API, requests
- **Environment**: kaggle-environments (multi-armed bandit simulator)

## Competition

| | |
|---|---|
| **Competition** | [Santa 2020 — The Candy Cane Contest](https://www.kaggle.com/competitions/santa-2020) |
| **Type** | Simulation (multi-armed bandit) |
| **Metric** | Leaderboard rank (agent vs. agent matchups) |
| **Timeline** | December 2020 -- January 2021 |
