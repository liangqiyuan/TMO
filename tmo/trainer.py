"""Single-experiment training/evaluation utilities."""

from tmo.evaluator import evaluate


def process_model(model_key, model_cls, train_env, test_env, latency_budget, usage_budget, results, device='auto'):
    """Train ``model_cls`` (if non-None) on ``train_env`` and evaluate on ``test_env``.

    Results are appended in-place to ``results[model_key]`` so that callers can
    accumulate runs across seeds. The signature is unchanged from the original
    open-source release except for an explicit ``device`` argument so that
    ``run_parallel`` can pin a specific GPU per worker.
    """
    if model_cls is None:
        results.setdefault(model_key, []).append(
            evaluate(env=test_env,
                     latency_budget=latency_budget,
                     usage_budget=usage_budget,
                     name=model_key[1])
        )
    else:
        model = model_cls('MlpPolicy', train_env, device=device)
        model.learn(total_timesteps=30000)
        results.setdefault(model_key, []).append(
            evaluate(env=test_env,
                     latency_budget=latency_budget,
                     usage_budget=usage_budget,
                     model=model,
                     name=model_key[1])
        )
