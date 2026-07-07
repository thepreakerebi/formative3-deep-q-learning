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
| `CnnPolicy` | Learns reliably: −20.7 → −17.7 (baseline config, exp 1); −13.9 at the tuned config (exp 3) |
| `MlpPolicy` | Never learns: flat −20.7 → −21.0 over 500k steps at the same tuned config (exp 9) |

**Why:** the MLP flattens the 84×84×4 stacked frames into a ~28k-dimensional vector, discarding
all spatial structure. The CNN's convolutions detect the ball and paddles as local patterns
wherever they appear on screen (translation invariance) and the frame stack gives it motion —
without that prior, raw pixels are unlearnable noise at this scale. MlpPolicy is only viable for
low-dimensional state inputs (e.g., RAM observations), never for pixels.

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
| 3 | lr=**5e-5**, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (CnnPolicy, 500k steps) | Best result so far — halving the baseline lr improved everything: breakout from the flat phase ~100k steps earlier (−19.5 by 200k vs baseline's 300k), then −16.3 by 300k, −13.9 mean over the final 100k with best episode −6. Episode length grew 3.5k→7.4k frames. Confirms 1e-4 was already slightly too hot: smoother Q-updates learn faster AND end better. lr picture is now monotonic across exps 1–3: 2.5e-4 fails, 1e-4 works, 5e-5 wins. |
| 4 | lr=**2.5e-5**, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (CnnPolicy, 500k steps) | Undershoots the budget: learning is visible but ~250k steps behind exp 3's pace — near-flat until 300k (−20.2), first real movement only in the final segment (−18.7, best −12), episode length 3.4k→5.3k. Same learning curve shape as exps 1/3, just too slow to converge in 500k steps. Completes a U-shaped lr sensitivity curve: 2.5e-4 diverges, 1e-4 works, **5e-5 optimal**, 2.5e-5 too slow. lr locked at 5e-5 for the remaining experiments. |
| 5 | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=**0.3** (CnnPolicy, 500k steps) | Extended exploration hurt: annealing epsilon over 150k steps (vs 50k) left the agent ~2–4 points behind exp 3 at every stage — −19.8 at 300k (exp 3: −16.3), final −18.0 vs −13.9. Longest episodes of any run (8.8k frames) but low reward: long rallies driven partly by residual random actions, not skill. Conclusion: exploration was never the bottleneck at this budget — the extra random actions fill the replay buffer with low-quality experience and delay exploitation. |
| 6 | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=**0.02** (CnnPolicy, 500k steps) | Second-best run. Minimal exploration (ε floors at 10k steps) did not prevent learning but delayed it: tracked ~60–80k steps behind exp 3's curve throughout, finishing at −15.5 in the final segment (best −10) vs exp 3's −13.9 — still climbing steeply at cutoff (episode length hit 12k frames, the longest of any run). Together with exps 3/5 this brackets the epsilon axis: 0.02 costs a delayed takeoff, 0.3 costs ~4 points, 0.1 is the sweet spot. Epsilon errors degrade gracefully — unlike lr errors, which are catastrophic. |
| 7 | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=**0.05** (CnnPolicy, 500k steps) | Surprisingly the weakest of the three epsilon variants (−18.5 final) despite sitting between two stronger configs (0.02 → −15.5, 0.1 → −13.9): very late breakout (~250k), climbed to −18.2 by 400k, then plateaued/regressed slightly (−18.7). Since a smooth epsilon landscape can't produce a dip between two peaks, the most plausible reading is **run-to-run variance** — breakout timing is partly luck, and a single 500k run per config can't fully separate schedule effects from noise. A key methodological caveat for interpreting all single-run results in this table. |
| 8 | lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=**0.1**, epsilon_decay=0.1 (CnnPolicy, 500k steps) | Chosen to test whether exp 5's harm came from anneal length or from randomness itself: a permanent 10% random-action floor. Prediction confirmed — earlier breakout than most runs (−19.6 by 200k, continued exploration keeps the buffer diverse) but a **capped ceiling**: final −17.7 vs −13.9 for the identical config with a 1% floor. With ε=0.1 forever, ~1 in 10 actions is random even when the network knows better — a persistent tax in a game where one missed ball costs a point. Confirms exploration should be front-loaded, then minimized. |
| 9 | **MlpPolicy**, lr=5e-5, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 (500k steps) | The required MLP vs CNN comparison, run at the champion config so the gap isolates architecture. Total failure: flat at −20.7 and drifting to −21.0 by the end (worse than random), episode length shrinking. The MLP flattens the 84×84×4 frame stack into a ~28k-dim vector, destroying the spatial structure (ball/paddle positions and motion) that a CNN's convolutions exploit — with no spatial prior, the pixel input is unlearnable noise at this scale. Also trained ~6× faster (749 vs ~130 it/s): cheap and useless. Verdict: **CnnPolicy is mandatory for pixel observations.** |
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
