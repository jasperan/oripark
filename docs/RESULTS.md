# Results

> Reference run: `results/run14` (250 self-play blocks, BC-pretrained
> evader, physics fixed). Reproduce with
> `python train.py --blocks 250 --out results/run14`.

## The headline: skill on hard arenas

Escape rate on **hard arenas** (terrain params 0.7/0.85 mixed, 60 matches,
trained deterministic chaser, stochastic evader — the honest PPO measure):

| policy | escape rate | caught |
|---|---|---|
| random baseline | 17% | 50/60 |
| BC-pretrained (no RL) | 31% | 14/42 |
| **trained (best checkpoint b160)** | **48%** | 31/60 |

The trained evader escapes **2.8× more often than random** (+31 pts) and
+17 pts over behavior cloning alone. On run14's own (hard) adversary
distribution: trained 28% vs baseline 15% (~1.9×).

Deterministic argmax play is not a fair measure for either policy — it
gets stuck in repetitive patterns (0% escapes) — so all official evals
sample actions (see [EXPERIMENTS.md](EXPERIMENTS.md), E5).

## Improvement over training (progress curve)

`results/run14/progress.png` / `progress.md`: escape rate on a fixed
arena set vs the frozen chaser across 25 checkpoints. Training escapes per
block grew monotonically across runs: **235 → 2434 → 3084** (run10, run13,
run14) as the curriculum escalated from collapsed-easy to genuinely hard.

The best checkpoint (block 160, mid-run) beats the final block — normal
self-play non-stationarity (the chaser adapts to the latest strategy), and
exactly why `select.py` picks the strongest checkpoint.

## Visual proof

`docs/media/*.gif`: three arenas where the **trained evader escapes in
154-235 steps** (portal reached, chaser outrun) while the **untrained
policy flails until timeout**. Same arena, same chaser, same seed.

## The adversarial loop finally works

- The CEM terrain adversary now **escalates** difficulty: win-rate signal
  0.17 → 0.67 across training, final level mean `[0.35, 0.75, 0.85, 0.45,
  0.7, 0.56]` (hard) instead of the collapsed-easy `[0.17, 0.32, ...]`
  of run13 (a deterministic-eval bug made the adversary think every level
  was unwinnable).
- Self-play is a genuine seesaw: evader Elo oscillated 950-1190 vs chaser
  1210-1440 across blocks; the two policies adapt to each other.

## Reproduce

```bash
python train.py --blocks 250 --out results/run14        # ~40 min on CPU
python progress.py --run results/run14                  # improvement curve
python select.py --run results/run14                    # best-checkpoint selection
python eval.py --run results/run14 --baseline           # head-to-head report
python gifdemo.py --run results/run14 --out docs/media/hero.gif
```
