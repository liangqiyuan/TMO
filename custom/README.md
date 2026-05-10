# Custom Scaling Sweep

## Use Case

This readme shows how to apply TMO to a multi-modal setup that differs from the M4A1 dataset — a different number of modalities, tasks, dialogue lengths, local hardware, or cloud endpoint — without touching anything inside `tmo/`. Everything runs on a fully synthetic dataset whose response scores, association scores, latencies, and costs are written out explicitly in [`run_custom.py`](./run_custom.py), so you can read off and replace every number with one from your own scenario.

## Run

```bash
# full sweep: 7 configs × {Local, Cloud, Random, A2C, RC-A2C} × 3 random seeds
python custom/run_custom.py --repeat 3
```

Results are written to `custom/results/Custom_Scaling.pkl`; rerun (overwriting the cache) whenever you change a flag. Each `(configuration, resource_constraint, seed)` cell trains the chosen RL policy for **30 000 timesteps**, matching the budget used by `tmo.process_model` in the main experiments. Both `Resource_Constraint=False` and `Resource_Constraint=True` variants of the RL policy are always evaluated side-by-side; the heuristic baselines (`Local`, `Cloud`, `Random`) are evaluated once because they ignore the resource-aware observation.

The seven `(num_modalities, num_tasks, num_turns)` configurations are listed in the `CONFIGS` constant at the top of [`run_custom.py`](./run_custom.py) — edit that list to add or remove a scaling point. CLI flags only cover the cross-cutting knobs that apply to every cell:

| Flag | Default | Purpose |
|---|---|---|
| `--policy` | `A2C` | RL backbone — one of `PPO`, `A2C`, `DQN`. |
| `--repeat` | `3` | Number of independent random seeds per cell. |
| `--num_gpus` | `1` | Round-robin GPU pool size for `tmo.run_parallel`. |
| `--num_episodes` | `10000` | Synthetic episodes generated per cell. |
| `--time_span` | `5` | Length of the RL sliding observation window. Independent of `num_turns`: when `time_span < num_turns` (e.g. the *Long History* config) the policy only sees the most recent `time_span` turns. |
| `--latency_budget` / `--usage_budget` | `30.0 s` / `0.05 USD` | Per-episode resource budgets. |
| `--alpha` / `--beta_association` / `--beta_latency` / `--beta_usage` | `1` / `1/3` / `1/3` / `1/3` | Reward weights `[α, β_a, β_l, β_u]`. |

## Scenario Design

Each turn in a synthetic episode is built around four user-controllable signals: the **response score**, the **association score**, the **latency**, and the **monetary cost**. They are kept deliberately simple so that any reward separation between policies has to come from the actual local–cloud trade-off rather than from a hand-tuned bias.

#### Response score

Per turn, an action is sampled with `cloud_prob = 0.5` (cloud) and `upload_prob = 0.7` (when cloud, upload a non-empty random subset of the available modalities; otherwise text-only). Conditioned on the action, the response score is drawn from a clipped Gaussian:

| Action | Distribution | Range |
|---|---|---|
| Local | `N(0.7, 0.3²)` | clipped to `[0, 2]` |
| Cloud (text-only or with uploads) | `N(1.0 + 0.1 · cum_uploaded, 0.3²)` | clipped to `[0, 2]` |

`cum_uploaded` is the *cumulative* number of modalities uploaded so far in the episode, incremented **before** the score is sampled. This encodes the intuition that cloud answers are systematically better than local ones, and that the model benefits monotonically — but with diminishing absolute return inside `[0, 2]` — from accumulating richer multi-modal context.

#### Association score

Each turn carries a length-`num_modalities` vector of per-modality association scores, drawn i.i.d. from `Uniform(0.1, 0.5)`. The environment's `Ass_transform` aggregates this vector over every candidate upload subset, so the per-action association score grows linearly in expectation with the subset size — observing more modalities can only help.

#### Local latency and cost

The local arm is computed analytically by `tmo.devices.compute_local_costs` from the **Jetson TX2** profile (`GF_peak = 1330 GFLOPS`, `P_max = 15 W`) and the default 3.8 B-parameter LLM at a 1024 → 2048 token budget, yielding `≈ 0.011 s` and `≈ 8 × 10⁻⁹ USD` per call (at `16.68 ¢/kWh`). In practice the local cost is effectively zero relative to the cloud arm; the reward separation comes entirely from the cloud channel.

#### Cloud latency and cost

A synthetic cloud profile is built per configuration by `synth_cloud_profile(num_modalities)`:

```
cloud_time[k]       = 0.5    + 1.5   · k     # seconds   (k = #uploaded modalities, 0 = text-only)
cloud_usage_cost[k] = 0.0005 + 0.004 · k     # USD
```

Both arrays are linear in `k` with length `num_modalities + 1`, so the policy faces a genuine trade-off: bigger uploads buy higher response and association scores but burn more latency and money.

#### Reward

```
reward_t  =  α · response_t  +  β_a · association_t  −  β_l · norm_latency_t  −  β_u · norm_usage_t
```

`norm_latency_t` and `norm_usage_t` are min–max normalisations of the length-`(num_modalities + 2)` cost vector `[local_cost, cloud_cost(k=0), …, cloud_cost(k=num_modalities)]`, so the local action always normalises to `0` and the largest cloud upload always normalises to `1` regardless of `num_modalities`.

## Results

Aggregated metrics across `--repeat 3` seeds with `--policy A2C`, reported as `mean ± std`. **Constraint Violation** columns measure how much each method *exceeds* the per-episode budgets (`30 s` latency, `0.05 USD` usage); a value of `0.00` means the budget was respected — non-zero violations are bolded. Action counts are summed across all evaluation episodes.

#### M4A1-shape — `num_modalities=3, num_tasks=4, num_turns=5`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.69 ± 0.00 | 0.06 ± 0.00 | 0.00 ± 0.00 | 0.69 ± 0.00 | 50000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.35 ± 0.00 | 13.76 ± 0.02 | 32.53 ± 0.05 | 1.16 ± 0.00 | 0 | 6233 | 43767 | 0.00 | **1.49** |
| Random | 0.97 ± 0.00 | 6.96 ± 0.04 | 16.38 ± 0.10 | 0.88 ± 0.00 | 24898 | 3063 | 22039 | 0.00 | **1.24** |
| TMO (A2C) | 1.57 ± 0.08 | 21.00 ± 3.08 | 51.83 ± 8.22 | 1.28 ± 0.04 | 0 | 0 | 50000 | 0.00 | **4.33** |
| **TMO (RC-A2C)** | 1.15 ± 0.11 | 7.50 ± 3.54 | 15.83 ± 9.43 | 1.04 ± 0.06 | 0 | 16667 | 33333 | 0.00 | 0.00 |

#### Small — `num_modalities=2, num_tasks=2, num_turns=2`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.71 ± 0.06 | 0.02 ± 0.00 | 0.00 ± 0.00 | 0.71 ± 0.06 | 20000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.15 ± 0.04 | 3.99 ± 0.01 | 8.98 ± 0.04 | 0.95 ± 0.04 | 0 | 5050 | 14950 | 0.00 | 0.00 |
| Random | 0.91 ± 0.05 | 2.03 ± 0.00 | 4.53 ± 0.01 | 0.81 ± 0.05 | 9907 | 2511 | 7582 | 0.00 | 0.00 |
| TMO (A2C) | 1.16 ± 0.04 | 3.00 ± 1.41 | 6.33 ± 3.77 | 1.00 ± 0.02 | 0 | 6667 | 13333 | 0.00 | 0.00 |
| **TMO (RC-A2C)** | 1.16 ± 0.04 | 2.75 ± 1.27 | 5.66 ± 3.40 | 1.01 ± 0.02 | 0 | 8360 | 11640 | 0.00 | 0.00 |

#### Medium — `num_modalities=4, num_tasks=4, num_turns=4`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.70 ± 0.02 | 0.05 ± 0.00 | 0.00 ± 0.00 | 0.70 ± 0.02 | 40000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.39 ± 0.00 | 14.01 ± 0.04 | 34.03 ± 0.09 | 1.21 ± 0.00 | 0 | 2512 | 37488 | 0.00 | **4.81** |
| Random | 1.00 ± 0.01 | 7.09 ± 0.05 | 17.17 ± 0.12 | 0.91 ± 0.01 | 19865 | 1209 | 18926 | 0.00 | **4.40** |
| TMO (A2C) | 1.72 ± 0.03 | 24.50 ± 1.22 | 62.00 ± 3.27 | 1.41 ± 0.01 | 0 | 0 | 40000 | 0.00 | **12.00** |
| **TMO (RC-A2C)** | 1.34 ± 0.15 | 12.00 ± 2.83 | 28.67 ± 7.54 | 1.18 ± 0.11 | 0 | 0 | 40000 | 0.00 | 0.00 |

#### Large — `num_modalities=6, num_tasks=6, num_turns=6`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.70 ± 0.01 | 0.07 ± 0.00 | 0.00 ± 0.00 | 0.70 ± 0.01 | 60000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.54 ± 0.00 | 30.01 ± 0.03 | 75.02 ± 0.09 | 1.36 ± 0.00 | 0 | 962 | 59038 | **4.09** | **25.45** |
| Random | 1.08 ± 0.00 | 15.10 ± 0.07 | 37.67 ± 0.19 | 0.99 ± 0.00 | 29942 | 468 | 29590 | **2.84** | **10.39** |
| TMO (A2C) | 1.84 ± 0.01 | 48.91 ± 1.54 | 125.43 ± 4.10 | 1.55 ± 0.01 | 0 | 0 | 60000 | **18.91** | **75.43** |
| **TMO (RC-A2C)** | 1.06 ± 0.01 | 12.00 ± 0.00 | 27.00 ± 0.00 | 0.98 ± 0.01 | 0 | 0 | 60000 | 0.00 | 0.00 |

#### More Modalities — `num_modalities=10, num_tasks=3, num_turns=3`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.68 ± 0.03 | 0.03 ± 0.00 | 0.00 ± 0.00 | 0.68 ± 0.03 | 30000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.66 ± 0.00 | 23.98 ± 0.04 | 61.45 ± 0.12 | 1.48 ± 0.00 | 0 | 29 | 29971 | **2.48** | **15.01** |
| Random | 1.14 ± 0.01 | 12.10 ± 0.06 | 30.96 ± 0.16 | 1.05 ± 0.01 | 14913 | 17 | 15070 | **2.32** | **12.07** |
| TMO (A2C) | 1.93 ± 0.05 | 34.50 ± 2.12 | 89.50 ± 5.66 | 1.69 ± 0.04 | 0 | 0 | 30000 | **4.50** | **39.50** |
| **TMO (RC-A2C)** | 1.71 ± 0.06 | 22.59 ± 1.11 | 57.75 ± 2.97 | 1.54 ± 0.05 | 0 | 0 | 30000 | 0.00 | **0.00** |

#### More Tasks — `num_modalities=3, num_tasks=10, num_turns=3`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.71 ± 0.01 | 0.03 ± 0.00 | 0.00 ± 0.00 | 0.71 ± 0.01 | 30000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.23 ± 0.00 | 8.24 ± 0.03 | 19.48 ± 0.07 | 1.04 ± 0.00 | 0 | 3749 | 26251 | 0.00 | 0.00 |
| Random | 0.95 ± 0.00 | 4.17 ± 0.01 | 9.81 ± 0.03 | 0.85 ± 0.00 | 14913 | 1860 | 13227 | 0.00 | 0.00 |
| TMO (A2C) | 1.42 ± 0.07 | 12.00 ± 2.12 | 29.50 ± 5.66 | 1.15 ± 0.02 | 0 | 0 | 30000 | 0.00 | 0.00 |
| **TMO (RC-A2C)** | 1.18 ± 0.17 | 6.50 ± 3.74 | 14.83 ± 9.98 | 1.02 ± 0.09 | 0 | 13333 | 16667 | 0.00 | 0.00 |

#### Long History — `num_modalities=3, num_tasks=3, num_turns=10`

| Method | Response ↑ | Latency (s) ↓ | Usage (10⁻³ USD) ↓ | Reward ↑ | Local | Cloud (text) | Cloud (+mod) | Lat. Viol. ↓ | Usage Viol. ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Local Only | 0.71 ± 0.02 | 0.07 ± 0.00 | 0.00 ± 0.00 | 0.71 ± 0.02 | 100000 | 0 | 0 | 0.00 | 0.00 |
| Cloud Only | 1.50 ± 0.00 | 16.50 ± 0.03 | 39.01 ± 0.08 | 1.31 ± 0.00 | 0 | 12477 | 87523 | 0.00 | **3.41** |
| Random | 1.04 ± 0.01 | 8.32 ± 0.03 | 19.60 ± 0.07 | 0.94 ± 0.01 | 49876 | 6206 | 43918 | 0.00 | **2.27** |
| TMO (A2C) | 1.61 ± 0.02 | 22.38 ± 5.74 | 54.69 ± 15.30 | 1.38 ± 0.04 | 0 | 7011 | 92989 | 0.00 | **9.00** |
| **TMO (RC-A2C)** | 0.99 ± 0.01 | 3.00 ± 0.00 | 3.00 ± 0.00 | 0.94 ± 0.01 | 0 | 100000 | 0 | 0.00 | 0.00 |

Across every configuration, **TMO (RC-A2C)** is the only method that *systematically* respects both budgets while remaining competitive on overall reward. Unconstrained TMO chases the highest possible response/association score and consequently overshoots the budget whenever the configuration permits it; the heuristic baselines violate the usage budget once the upload arrays grow.

## Adapting to Your Own Scenario

To port this template to a real scenario, only the four scenario-defining signals need to be replaced — everything else (environment, RL training, evaluation, plotting) is reused as-is.

**1. Dataset.** Provide your own dialogues in place of the synthetic ones. You need a per-turn **response score** (any quality metric of the model's answer; e.g. an LLM-as-judge rating or a task-specific score) and a per-turn **association score** between the user prompt and each modality (e.g. a CLIP-style relevance score). How exactly these are collected is scenario-specific and outside the scope of this template.

**2. Cloud profile.** Replace the synthetic latency / cost arrays with measurements from your own cloud endpoint and its published per-call pricing. The arrays must cover every possible upload count from `0` (text-only) up to `num_modalities`.

**3. Local profile.** Replace the on-device specs (peak compute throughput and peak power) with those of your target device. The local latency and energy cost are derived analytically from these numbers and your LLM's parameter / token budget.

**4. Configurations and budgets.** Edit `CONFIGS` to the `(num_modalities, num_tasks, num_turns)` tuples you care about; the observation and action spaces of `M4A1_Env` adapt automatically. Rescale the latency / usage budgets and reward weights so the response-score channel is comparable in magnitude to the normalised cost channels on your data.
