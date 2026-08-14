# Results

> Numbers below are from the reference run (`results/run10`). Re-run with
> `python train.py --blocks 150 --out results/run10` to reproduce; results
> vary run-to-run because self-play is a noisy seesaw.

## What improved

The trained evader (150 self-play blocks, ~2.5 M timesteps per side) was
evaluated on a **fixed set of arenas** sampled from the final
terrain-adversary distribution, against the **frozen, deterministic
chaser**, with the evader playing stochastically (the honest measure for a
PPO policy — deterministic argmax play gets stuck in repetitive patterns).

See `results/run10/progress.png` / `progress.md` for the full curve.

| metric | random baseline | early (b10) | mid (b100) | peak (b130) | final |
|---|---|---|---|---|---|
| escape rate | 0% | 5% | 23% | **54%** | **30%** |
| caught | 0 | 1 | 8 | 11 | 11 |

Takeaways:

- The **random baseline never escapes** (0/20) — it wanders near the
  portal but never completes the escape routine.
- Escape skill emerges around block 80-100 and peaks at **54%** (block
  130) with short, action-packed episodes.
- The final policy (30%) sits below the peak — normal self-play
  non-stationarity: the chaser adapts to the evader's strategy. This is
  exactly why the league keeps training.
- During training, avg traversal zone per episode rises from ~4 to
  ~10-12 of 15, and dash/wall-jump/double-jump/bash usage per episode
  climbs steadily (e.g. dash 2 → 6+, bash 0 → 3+ per episode).

## Training dynamics (self-play seesaw)

Self-play Elo oscillates as the two policies adapt to each other:

- Early blocks: the chaser learns to catch faster than the evader learns to
  escape (pursuer task is structurally easier), so chaser Elo pulls ahead.
- Mid-run: the evader's traversal improves (avg max zone per episode rises
  from ~4 to ~10-12 of 15), escapes start appearing (up to 6+ per block).
- Late-run: the evader survives longer, gets caught less, and the Elo gap
  closes (final evader ≈ 1140 vs chaser ≈ 1260 in the reference run).

The terrain adversary (CEM) starts at easy level parameters and hardens
only as the evader's win rate rises — with an "easy reset" if the whole
candidate population is unwinnable, so the curriculum never spirals into
impossible levels.

## Reward engineering timeline (what actually mattered)

1. Survival rewards made timeouts attractive → escape-oriented rebalance
   (portal +3, timeout −3, survival ≈ 0).
2. "Flee chaser" conflicted with "reach portal" (chaser spawns between) →
   portal-progress + rightward milestone rewards made the escape objective
   dominant.
3. The evader stalled at the chaser's spawn → one-time "pass the chaser"
   bonus.
4. The evader never learned to traverse against a strong chaser → warmup
   phase vs random opponents + asymmetric capacity (protagonist net/lr >
   adversary).
5. Timeouts polluted the win-rate metric → evals now report escape rate,
   catch rate, traversal zone, and survival separately.

Full details in [TUNING.md](TUNING.md).

## Reproduce

```bash
python train.py --blocks 150 --out results/run10        # ~25 min on CPU
python progress.py --run results/run10                  # improvement curve
python eval.py --run results/run10 --baseline           # head-to-head report
python demo.py --run results/run10 --mode both --out demo.gif
```
