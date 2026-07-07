# Formative 3: Deep Q Learning — Pong (ALE/Pong-v5)

Training and evaluating a Deep Q-Network (DQN) agent on Atari Pong using
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/) and
[Gymnasium](https://gymnasium.farama.org/).

**Course:** Machine Learning Techniques II — 2026 May Term (ALU)

**Team:**
- Elvis Preye Kerebi
- Glory Paul
- Leny Pascal

## Environment

[`ALE/Pong-v5`](https://ale.farama.org/environments/pong/) — the agent controls
the right paddle; reward is **+1** for each point scored and **−1** for each
point conceded (first to 21). Observations are raw RGB frames, preprocessed by
the standard Atari pipeline (grayscale, 84×84 resize, 4-frame stack) so the
network can perceive ball velocity.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training (`train.py`)

All assignment hyperparameters are CLI flags, so every team member runs their
tuning experiments with the same script:

```bash
python train.py --run-name elvis-exp1 --policy CnnPolicy \
  --lr 1e-4 --gamma 0.99 --batch-size 32 \
  --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 0.1 \
  --timesteps 200000
```

- Saves the trained model as **`dqn_model.zip`** (`--save-as` to override).
- Logs **reward trends** and **episode lengths** to `logs/<run-name>/monitor/`
  (CSV) and TensorBoard (`rollout/ep_rew_mean`, `rollout/ep_len_mean`).
- Writes the exact config of each run to `logs/<run-name>/config.json` for the
  experiment table.

### MLP vs CNN policy

| Policy | Result |
|---|---|
| `CnnPolicy` | Baseline (exp 1): −20.7 → −17.7 mean reward over 500k steps, stable learning curve |
| `MlpPolicy` | _to be filled after comparison run_ |

_Discussion of which policy performs better on Pong and why: to be added._

## Playing (`play.py`)

```bash
python play.py --model dqn_model.zip --episodes 3
```

Loads `dqn_model.zip`, recreates the same environment/preprocessing as
training with `render_mode="human"` (GUI display), and selects actions with
the **greedy Q policy** (`deterministic=True` — always the highest Q-value
action, no exploration).

## Gameplay Video

_Video of the agent playing (play.py running with the agent interacting with
the environment) — to be added._

## Hyperparameter Tuning Experiments

Each member ran 10 experiments. Epsilon maps to SB3 as: `epsilon_start` →
`exploration_initial_eps`, `epsilon_end` → `exploration_final_eps`,
`epsilon_decay` → `exploration_fraction`.

### MEMBER NAME: Elvis Preye Kerebi

| # | Hyperparameter Set | Noted Behavior |
|---|---|---|
| 1 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (baseline, CnnPolicy, 500k steps) | Flat at −20.7 for the first ~200k steps (exploration + replay warm-up), then steady learning: −19.5 by 300k, −17.7 mean by 500k with best episode −13. Episode length grew 3.5k→8.3k frames (rallies ~2.4× longer). Stable, no divergence; reward still climbing at cutoff, so longer training would keep improving. |
| 2 | lr=**2.5e-4**, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (CnnPolicy, 500k steps) | Complete failure to learn: mean reward flat at −20.9 across all 500k steps, best episode only −20 after 100k, episode length stuck at ~3.2k frames (vs baseline's growth to 8.3k). The 2.5× higher lr destabilized Q-value bootstrapping — updates chase a moving target and no coherent policy forms. Slightly worse than random by the end (degenerate policy). Conclusion: lr is the most sensitive knob so far; viable range is at or below ~1e-4. |
| 3 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 4 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 5 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 6 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 7 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 8 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 9 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 10 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |

### MEMBER NAME: Glory Paul

| # | Hyperparameter Set | Noted Behavior |
|---|---|---|
| 1 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 2 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 3 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 4 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 5 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 6 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 7 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 8 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 9 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 10 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |

### MEMBER NAME: Leny Pascal

| # | Hyperparameter Set | Noted Behavior |
|---|---|---|
| 1 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 2 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 3 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 4 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 5 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 6 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 7 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 8 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 9 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |
| 10 | lr= , gamma= , batch= , epsilon_start= , epsilon_end= , epsilon_decay= | |

## Results Discussion

_Discussion of the hyperparameter tuning results based on the tables above
(which changes improved performance, which harmed it, and the best final
configuration) — to be added once experiments are complete._
