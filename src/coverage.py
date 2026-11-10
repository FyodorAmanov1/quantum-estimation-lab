"""Conditional interval diagnostics for the studio's bounded exposure oracle."""
from functools import lru_cache

import numpy as np
from scipy.special import xlogy
from scipy.stats import beta


POWERS = (0, 1, 2, 4, 8)
SHOTS = 256
LR_DROP = 1.920729  # Existing nominal 95% chi-square(1) likelihood cutoff / 2.


def exact_binomial_interval(successes, total):
    """Equal-tailed 95% Clopper-Pearson interval, including both endpoints."""
    return [float(beta.ppf(.025, successes, total - successes + 1)) if successes else 0.,
            float(beta.ppf(.975, successes + 1, total - successes)) if successes < total else 1.]


@lru_cache(maxsize=1)
def _mesh():
    theta = np.linspace(1e-6, np.pi / 2 - 1e-6, 32769)
    p = np.clip(np.sin((2 * np.array(POWERS)[:, None] + 1) * theta) ** 2,
                1e-12, 1 - 1e-12)
    return theta, np.log(p), np.log1p(-p)


def likelihood_diagnostics(successes, truth):
    """Replay the legacy mesh and an endpoint-inclusive variant on shared counts.

    Interior probabilities retain the legacy numerical clipping. Endpoint
    likelihoods are exact, with 0*log(0)=0, rather than clipped probabilities.
    Membership at truth is evaluated directly, separately from the mesh hull.
    """
    successes = np.asarray(successes)
    if (successes.shape != (5,) or not np.issubdtype(successes.dtype, np.integer)
            or np.any((successes < 0) | (successes > SHOTS)) or not 0 <= truth <= 1):
        raise ValueError("Five integer counts in [0,256] and amplitude in [0,1] required")
    theta, logp, logq = _mesh()
    ll = np.zeros(len(theta))
    for i, s in enumerate(successes):
        ll += s * logp[i] + (SHOTS - s) * logq[i]
    true_p = np.sin((2 * np.array(POWERS) + 1) * np.arcsin(np.sqrt(truth))) ** 2
    true_p = np.clip(true_p, 1e-12, 1 - 1e-12)
    true_ll = float(np.sum(xlogy(successes, true_p) + xlogy(SHOTS - successes, 1 - true_p)))
    if truth in (0., 1.):
        true_ll = 0. if np.all(successes == SHOTS * truth) else -np.inf
    results = {}
    for name, endpoints in (("legacy", False), ("endpoint_inclusive", True)):
        values, angles = ll, theta
        if endpoints:
            values = np.r_[0. if np.all(successes == 0) else -np.inf,
                           ll, 0. if np.all(successes == SHOTS) else -np.inf]
            angles = np.r_[0., theta, np.pi / 2]
        best = int(np.argmax(values))
        accepted = values >= values[best] - LR_DROP
        bounds = np.sin(angles[accepted]) ** 2
        interval = [float(bounds.min()), float(bounds.max())]
        components = int(np.count_nonzero(accepted & ~np.r_[False, accepted[:-1]]))
        results[name] = {"estimate": float(np.sin(angles[best]) ** 2), "interval": interval,
                         "covered": bool(interval[0] <= truth <= interval[1]),
                         "likelihood_set_covered": bool((endpoints or 0 < truth < 1)
                                                        and true_ll >= values[best] - LR_DROP),
                         "components": components}
    return results


def _summary(runs, method, truth):
    observations = [r[method] for r in runs]
    errors = np.array([r["estimate"] - truth for r in observations])
    covered = sum(r["covered"] for r in observations)
    result = {"covered_count": covered, "empirical_coverage95": covered / len(runs),
              "coverage_mc_interval95": exact_binomial_interval(covered, len(runs)),
              "bias": float(errors.mean()), "rmse": float(np.sqrt(np.mean(errors**2))),
              "mean_interval_width": float(np.mean([r["interval"][1] - r["interval"][0]
                                                    for r in observations]))}
    if method != "classical_exact":
        result.update(likelihood_set_covered_count=sum(r["likelihood_set_covered"] for r in observations),
                      disconnected_count=sum(r["components"] > 1 for r in observations))
    return result


def amplitude_coverage_comparison(bounded=None, n_trials=128, seed=2505):
    """Independent fixed-schedule trials; no cutoff tuning using these trials."""
    try:
        from .studio import amplitude_oracle, bounded_exposure_comparison
    except ImportError:
        from studio import amplitude_oracle, bounded_exposure_comparison

    if type(n_trials) is not int or n_trials < 2 or type(seed) is not int or seed < 0:
        raise ValueError("At least two trials and a nonnegative integer seed required")
    if bounded is None:
        bounded = bounded_exposure_comparison()
    targets = [(f"a={a:g}", a, None) for a in
               (0., 1e-6, .001, .01, .5, .99, .999, 1 - 1e-6, 1.)]
    targets += [(r["model"], r["grid"]["amplitude"], r["grid"])
                for r in bounded["rows"]]
    branches = iter(np.random.SeedSequence(seed).spawn(len(targets) * n_trials * 2))
    queries = sum(SHOTS * (2 * k + 1) for k in POWERS)
    rows = []
    for label, truth, grid in targets:
        probabilities, payoff = ([1., 0.], [truth, truth]) if grid is None else (
            grid["probabilities"], grid["normalized_payoff"])
        state, Q = amplitude_oracle(probabilities, payoff)
        # One explicit unitary check per target; repeated trials sample the
        # identical ideal binomial law without repeatedly multiplying matrices.
        actual = [float(np.sum((np.linalg.matrix_power(Q, k) @ state)[1::2] ** 2)) for k in POWERS]
        analytic = np.sin((2 * np.array(POWERS) + 1) * np.arcsin(np.sqrt(truth))) ** 2
        if not np.allclose(actual, analytic, atol=1e-10, rtol=0):
            raise AssertionError("Explicit oracle differs from analytic measurement law")
        analytic = np.clip(analytic, 0, 1)
        if truth in (0., 1.):
            analytic[:] = truth
        runs = []
        for trial in range(n_trials):
            ae_seed, mc_seed = [int(next(branches).generate_state(1, dtype=np.uint64)[0]) for _ in range(2)]
            counts = np.random.default_rng(ae_seed).binomial(SHOTS, analytic)
            results = likelihood_diagnostics(counts, truth)
            count = int(np.random.default_rng(mc_seed).binomial(queries, truth))
            interval = exact_binomial_interval(count, queries)
            results["classical_exact"] = {"estimate": count / queries, "interval": interval,
                "covered": bool(interval[0] <= truth <= interval[1])}
            runs.append({"trial": trial, "measurement_seed": ae_seed, "classical_seed": mc_seed,
                         "successes_by_power": counts.tolist(), "classical_successes": count, **results})
        rows.append({"target": label, "amplitude": truth,
                     "encoded_mean_usd": None if grid is None else truth * grid["cap_usd"],
                     "explicit_probabilities": actual, "measurement_probabilities": analytic.tolist(),
                     "methods": {m: _summary(runs, m, truth) for m in
                                 ("legacy", "endpoint_inclusive", "classical_exact")}, "runs": runs})
    return {"n_trials": n_trials, "root_seed": seed, "nominal_coverage": .95,
            "powers": list(POWERS), "shots_per_power": SHOTS,
            "shots_per_trial": SHOTS * len(POWERS), "queries_per_trial": queries,
            "total_amplified_preparation_calls": queries * n_trials * len(targets),
            "total_classical_preparation_calls": queries * n_trials * len(targets),
            "as_of": bounded["as_of"], "data_source": bounded["data_source"],
            "seed_protocol": "SeedSequence(root).spawn(targets * trials * 2), target/trial/method order, uint64 child seeds; separate amplitude and classical streams. Legacy and endpoint-inclusive inference share counts.",
            "interval_method": "Legacy 32769-point open theta mesh; endpoint-inclusive adds exact theta=0,pi/2. Same fixed log-likelihood drop 1.920729. Report hull and direct likelihood-set membership separately. Classical baseline: exact equal-tailed Clopper-Pearson, 8960 power-zero shots.",
            "cost_model": "Fixed powers 0,1,2,4,8 with 256 shots each: 1280 shots, 8960 preparation/inverse calls. Classical baseline uses 8960 independent power-zero preparations. Logical query equality excludes preprocessing, inference, gates, noise and runtime.",
            "limitations": [
                "Conditional on fixed empirical capped/grid exposure, not coverage of uncapped population exposure, PFE or VaR. No new scenario paths or parameter fits per measurement trial.",
                "Asymptotic likelihood cutoff has no finite-sample coverage guarantee, especially at boundaries. Adding endpoints repairs support exclusion but does not calibrate coverage. The original estimator remains unchanged.",
                "Finite mesh can miss modes or set edges; hull can bridge excluded gaps. Direct truth membership uses the mesh maximum and is also numerical. No interpolated edges or mesh-convergence guarantee.",
                "Coverage uncertainty intervals describe finite independent pseudorandom trials per target, with no multiple-comparison correction. Bias and RMSE are also finite-trial estimates.",
                "Synthetic bundled data and ideal noiseless binomial measurements. Dense oracle and exact histogram mean already require classical answer-bearing preprocessing; no hardware or quantum advantage follows."],
            "rows": rows}


def amplitude_coverage_case(c):
    methods = ("legacy", "endpoint_inclusive", "classical_exact")
    return {"id": "amplitude-coverage", "label": "Amplitude interval coverage",
            "kicker": "BOUNDED EXPOSURE / REPEATED MEASUREMENT",
            "title": "Do likelihood intervals cover their target?",
            "description": f"{c['n_trials']} independent trials per amplitude, including exact and near boundaries and both option-book exposure oracles. Fixed query schedule; nominal 95% conditional intervals.",
            "metrics": [ {"label": "Trials per amplitude", "value": c["n_trials"]},
                         {"label": "Amplitude targets", "value": len(c["rows"])},
                         {"label": "Preparation calls per trial", "value": c["queries_per_trial"]},
                         {"label": "Nominal coverage", "value": 95, "unit": "%"}],
            "charts": [{"title": "Measured interval coverage (percent)",
                        "x": [r["target"] for r in c["rows"]], "xlabel": "Fixed amplitude / exposure model",
                        "series": [{"name": m, "values": [100 * r["methods"][m]["empirical_coverage95"] for r in c["rows"]]} for m in methods]
                                  + [{"name": "Nominal 95%", "values": [95] * len(c["rows"])}]},
                       {"title": "Mean conditional interval width (amplitude)",
                        "x": [r["target"] for r in c["rows"]], "xlabel": "Fixed amplitude / exposure model",
                        "series": [{"name": m, "values": [r["methods"][m]["mean_interval_width"] for r in c["rows"]]} for m in methods]}],
            "table": {"columns": ["Target", "Amplitude", "Method", "Covered / trials", "Coverage %", "MC 95% low %", "MC 95% high %", "Bias", "RMSE", "Mean width"],
                      "rows": [[r["target"], r["amplitude"], m, f"{s['covered_count']}/{c['n_trials']}",
                                100*s["empirical_coverage95"], *[100*x for x in s["coverage_mc_interval95"]],
                                s["bias"], s["rmse"], s["mean_interval_width"]]
                               for r in c["rows"] for m, s in r["methods"].items()]},
            "method": [c["seed_protocol"], c["interval_method"], c["cost_model"],
                       "Book targets are the existing GBM and variance-mixture empirical grid means divided by 400 USD. Multiply amplitude error and interval width by 400 for conditional USD units. Coverage does not change under this rescaling.",
                       "Exported JSON records all seeds, counts, estimates, intervals, likelihood-set membership and disconnected-component counts; table reports interval-hull coverage. MC coverage bounds are exact binomial intervals across trials."],
            "limitations": c["limitations"], "comparison": c,
            "reproduce": "python src/studio.py --out docs/study.json"}
