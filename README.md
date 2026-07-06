# Formative 3: Deep Q Learning — Pong (ALE/Pong-v5)

Training and evaluating a Deep Q-Network (DQN) agent on Atari Pong using
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/) and
[Gymnasium](https://gymnasium.farama.org/).

**Course:** Machine Learning Techniques II — 2026 May Term (ALU)

**Team:**
- Elvis Preye Kerebi
- Glory Paul
- Leny Pascal

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train (hyperparameters are CLI-configurable — see `python train.py --help`):

```bash
python train.py --policy CnnPolicy --lr 1e-4 --gamma 0.99 --batch-size 32 \
  --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 0.1 --timesteps 1000000
```

Play with the trained agent (greedy Q policy, GUI rendering):

```bash
python play.py --model dqn_model.zip --episodes 3
```

_Full documentation (hyperparameter experiment tables, results discussion, and
gameplay video) is added as the project progresses — see TEAM_TASKS.md for the
work plan._
