"""Replayable real-state preparation and an explicitly scoped logical gate ledger."""
from collections import Counter
from numbers import Integral

import numpy as np


def preparation_circuit(probabilities, payoff):
    """Binary probability tree, compiled with cyclic Gray-code multiplexed Ry gates.

    Qubit zero is most significant; the payoff ancilla is least significant.
    All rotations, including zero angles, are retained. No gate cancellation.
    """
    p, f = np.asarray(probabilities, float), np.asarray(payoff, float)
    if (p.ndim != 1 or f.shape != p.shape or len(p) < 2
            or len(p) & (len(p) - 1) or not np.isfinite(p).all()
            or not np.isfinite(f).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
            or np.any((f < 0) | (f > 1))):
        raise ValueError("Finite normalized power-of-two probabilities and bounded payoffs required")
    joint = np.column_stack((p * (1 - f), p * f)).ravel()
    n = int(np.log2(len(joint)))
    gates = []
    for target in range(n):
        masses = joint.reshape(2**target, 2, -1).sum(axis=2)
        angles = 2 * np.arctan2(np.sqrt(masses[:, 1]), np.sqrt(masses[:, 0]))
        size = len(angles)
        gray = [j ^ (j >> 1) for j in range(size)]
        # Inverse Walsh transform with Gray-code column order.
        beta = [sum((-1)**((b & g).bit_count()) * angles[b]
                    for b in range(size)) / size for g in gray]
        for j, angle in enumerate(beta):
            gates.append({"gate": "Ry", "target": target, "angle": float(angle)})
            if target:
                changed = gray[j] ^ gray[(j + 1) % size]
                control = target - changed.bit_length()
                gates.append({"gate": "CNOT", "control": control, "target": target})
    return {"data_qubits": n, "gates": gates}


def inverse_circuit(gates):
    return [dict(g, angle=-g["angle"]) if g["gate"] == "Ry" else dict(g)
            for g in reversed(gates)]


def replay(gates, state):
    """Apply the emitted gates to a vector or columns of vectors (MSB convention)."""
    result = np.array(state, copy=True, dtype=float)
    n = int(np.log2(len(result)))
    if len(result) != 2**n:
        raise ValueError("Power-of-two state dimension required")
    indices = np.arange(len(result))
    for g in gates:
        bit = 1 << (n - 1 - g["target"])
        name = g["gate"]
        if name in {"Z", "CZ"}:
            selected = (indices & bit) != 0
            if name == "CZ":
                selected &= (indices & (1 << (n - 1 - g["control"]))) != 0
            result[selected] *= -1
            continue
        selected = (indices & bit) == 0
        controls = ([g["control"]] if name == "CNOT" else
                    g["controls"] if name == "CCX" else [])
        for control in controls:
            selected &= (indices & (1 << (n - 1 - control))) != 0
        low = indices[selected]
        high = low | bit
        a, b = result[low].copy(), result[high].copy()
        if name == "Ry":
            c, s = np.cos(g["angle"] / 2), np.sin(g["angle"] / 2)
            result[low], result[high] = c * a - s * b, s * a + c * b
        elif name in {"X", "CNOT", "CCX"}:
            result[low], result[high] = b, a
        else:
            raise ValueError("Unsupported gate")
    return result


def zero_reflection_circuit(data_qubits):
    """I - 2|0><0| using a clean AND ladder, then uncomputing its workspace.

    CCX (Toffoli) is an explicitly retained three-qubit primitive, not a CNOT.
    """
    if not isinstance(data_qubits, Integral) or isinstance(data_qubits, bool) or data_qubits < 2:
        raise ValueError("At least two data qubits required")
    n = int(data_qubits)
    wrap = [{"gate": "X", "target": j} for j in range(n)]
    ladder = []
    for j in range(n - 2):
        controls = [0, 1] if j == 0 else [n + j - 1, j + 1]
        ladder.append({"gate": "CCX", "controls": controls, "target": n + j})
    control = 0 if n == 2 else 2 * n - 3
    phase = {"gate": "CZ", "control": control, "target": n - 1}
    return {"work_qubits": n - 2, "gates": wrap + ladder + [phase] + list(reversed(ladder)) + wrap}


def schedule_ledger(preparation, counts):
    """Serial, unoptimized gate inventory in {Ry,CNOT,X,Z,CZ,CCX}."""
    n = preparation["data_qubits"]
    reflection = zero_reflection_circuit(n)
    a = Counter(g["gate"] for g in preparation["gates"])
    s0 = Counter(g["gate"] for g in reflection["gates"])
    rows = []
    for row in counts:
        k, shots = row["grover_power"], row["shots"]
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (k, shots)) or k < 0 or shots < 1:
            raise ValueError("Nonnegative integer powers and positive integer shots required")
        forward, inverse, grover = shots * (k + 1), shots * k, shots * k
        gates = {g: (forward + inverse) * a[g] + grover * s0[g]
                 + (grover if g == "Z" else 0)
                 for g in ("Ry", "CNOT", "X", "Z", "CZ", "CCX")}
        rows.append({"grover_power": int(k), "shots": int(shots),
                     "preparation_calls": int(forward), "inverse_preparation_calls": int(inverse),
                     "grover_iterations": int(grover), "zero_reflections": int(grover),
                     "good_reflections": int(grover), "controlled_A_calls": 0,
                     "gates": gates})
    totals = {key: sum(r[key] for r in rows) for key in (
        "shots", "preparation_calls", "inverse_preparation_calls", "grover_iterations",
        "zero_reflections", "good_reflections", "controlled_A_calls")}
    totals["gates"] = {g: sum(r["gates"][g] for r in rows) for g in ("Ry", "CNOT", "X", "Z", "CZ", "CCX")}
    totals["preparation_or_inverse_calls"] = totals["preparation_calls"] + totals["inverse_preparation_calls"]
    return {"data_qubits": n, "clean_work_qubits": reflection["work_qubits"],
            "peak_qubits": n + reflection["work_qubits"], "per_preparation": dict(a),
            "per_zero_reflection": dict(s0), "rows": rows, "totals": totals}


def oracle_compilation_comparison(bounded):
    """Compile the existing empirical exposure laws and compare exact state actions."""
    try:
        from .studio import amplitude_oracle
    except ImportError:
        from studio import amplitude_oracle
    rows = []
    for source in bounded["rows"]:
        grid = source["grid"]
        circuit = preparation_circuit(grid["probabilities"], grid["normalized_payoff"])
        state, Q = amplitude_oracle(grid["probabilities"], grid["normalized_payoff"])
        A = replay(circuit["gates"], np.eye(len(state)))
        S0 = np.eye(len(state)); S0[0, 0] = -1
        good = np.diag(np.tile([1., -1.], len(state) // 2))
        compiled_Q = -A @ S0 @ A.T @ good
        counts = source["amplitude_estimation"]["counts"]
        ledger = schedule_ledger(circuit, counts)
        queries = ledger["totals"]["preparation_or_inverse_calls"]
        baseline = schedule_ledger(circuit, [{"grover_power": 0, "shots": queries}])
        probabilities = []
        for row in counts:
            k = row["grover_power"]
            actual = float(np.sum((np.linalg.matrix_power(compiled_Q, k) @ A[:, 0])[1::2]**2))
            dense = float(np.sum((np.linalg.matrix_power(Q, k) @ state)[1::2]**2))
            probabilities.append({"grover_power": k, "compiled": actual, "householder": dense,
                                  "absolute_difference": abs(actual - dense)})
        rows.append({"model": source["model"], "levels": grid["levels"],
                     "exact_grid_mean_usd": grid["grid_mean_usd"], "preparation_circuit": circuit,
                     "zero_reflection_circuit": zero_reflection_circuit(circuit["data_qubits"]),
                     "ledger": ledger, "power_zero_baseline": baseline,
                     "validation": {"state_max_abs_error": float(np.max(np.abs(A[:, 0] - state))),
                                    "inverse_max_abs_error": float(np.max(np.abs(replay(inverse_circuit(circuit["gates"]), A) - np.eye(len(state))))),
                                    "grover_matrix_max_abs_error": float(np.max(np.abs(compiled_Q - Q))),
                                    "success_probabilities": probabilities}})
    sensitivity = []
    for levels in (16, 32, 64):
        circuit = preparation_circuit(np.full(levels, 1 / levels), np.linspace(0, 1, levels))
        ledger = schedule_ledger(circuit, bounded["rows"][0]["amplitude_estimation"]["counts"])
        errors = [s for s in bounded["rows"][0]["sensitivity"] if s["levels"] == levels and s["cap_usd"] == 400][0]
        sensitivity.append({"levels": levels, "grid_error_usd": errors["grid_error_usd"],
                            "per_preparation": ledger["per_preparation"], "peak_qubits": ledger["peak_qubits"],
                            "schedule_gates": ledger["totals"]["gates"]})
    return {"as_of": bounded["as_of"], "data_source": bounded["data_source"], "rows": rows,
            "grid_sensitivity_gbm": sensitivity,
            "conventions": [
                "Q = -A S0 A^-1 Sgood; time order Sgood, inverse A, S0, A. Initial A is counted for every shot. Global minus is free; Q powers are repeated, not controlled.",
                "A is a binary joint-probability tree with cyclic Gray-code Ry/CNOT multiplexors. No zero-angle pruning, cross-block cancellation or connectivity routing. Ry angles are exact real numbers for this logical model.",
                "For L exposure nodes and n=log2(2L) data qubits, unpruned A has 2L-1 Ry and 2L-2 CNOT. Grid-size sensitivity counts depend only on circuit shape; its USD errors reuse the actual GBM cap/grid experiment.",
                "Sgood is Z on the payoff ancilla. S0 uses X on all data qubits, a clean AND ladder of CCX, CZ, the reversed ladder and X again. Work qubits start and end at zero and are reused.",
                "CCX is a three-qubit primitive retained separately. The ledger is mixed-level, not an elementary two-qubit total or a fault-tolerant T count. Arbitrary rotation synthesis precision, noise, routing, reset, runtime and classical inference are excluded.",
                "Power-zero baseline uses the same A and 8960 shots, matching preparation/inverse calls, not gates or runtime. Direct classical histogram summation already knows the mean and needs no quantum gates.",
            ], "limitations": [
                "This compiles the already computed empirical histogram, not coherent scenarios, option pricing or CSV loading. Classical preparation costs from the bounded exposure case remain payable.",
                "A differs from the Householder unitary away from |0>, but A S0 A^-1 and all Grover success probabilities agree numerically. This is an alternative compilation of the state action, not a decomposition of the full Householder matrix.",
                "Synthetic prices and ideal statevectors; no hardware performance, market accuracy or quantum advantage claim. Cap, grid, sampling and model errors remain separate.",
            ]}


def oracle_compilation_case(comparison):
    c = comparison
    t = c["rows"][0]["ledger"]["totals"]
    return {"id": "oracle-compilation", "label": "Oracle compilation ledger",
            "kicker": "BOUNDED EXPOSURE / LOGICAL GATE INVENTORY",
            "title": "What is hidden inside a preparation call?",
            "description": f"As of {c['as_of']}. {c['data_source']}. Replayable preparation of both 32-node exposure laws; separate forward, inverse and controlled gates.",
            "metrics": [{"label": key, "value": value, "unit": "calls"} for key, value in (
                ("Forward preparations", t["preparation_calls"]), ("Inverse preparations", t["inverse_preparation_calls"]),
                ("CNOT gates", t["gates"]["CNOT"]), ("Toffoli gates (CCX)", t["gates"]["CCX"]))],
            "charts": [{"title": "Same logical call budget: compiled gates per model",
                        "x": list(t["gates"]), "xlabel": "Gate type; CCX retained as a three-qubit primitive",
                        "series": [{"name": label, "values": list(ledger["totals"]["gates"].values())}
                                   for label, ledger in (("Amplitude schedule", c["rows"][0]["ledger"]),
                                                         ("Power-zero sampling", c["rows"][0]["power_zero_baseline"]))]},
                       {"title": "Grid precision versus preparation size (400 USD cap, GBM)",
                        "x": [r["levels"] for r in c["grid_sensitivity_gbm"]], "xlabel": "Exposure grid levels",
                        "series": [{"name": "Ry gates per A", "values": [r["per_preparation"]["Ry"] for r in c["grid_sensitivity_gbm"]]},
                                   {"name": "CNOT per A", "values": [r["per_preparation"]["CNOT"] for r in c["grid_sensitivity_gbm"]]}]}],
            "table": {"columns": ["Power", "Shots", "A", "Inverse A", "S0", "Sgood", "Ry", "CNOT", "CCX", "CZ"],
                      "rows": [[r["grover_power"], r["shots"], r["preparation_calls"], r["inverse_preparation_calls"],
                                r["zero_reflections"], r["good_reflections"], *[r["gates"][g] for g in ("Ry", "CNOT", "CCX", "CZ")]]
                               for r in c["rows"][0]["ledger"]["rows"]]},
            "method": c["conventions"] + [
                f"{r['model']}: state error {r['validation']['state_max_abs_error']:.3g}; Grover matrix error {r['validation']['grover_matrix_max_abs_error']:.3g}. Six data plus four clean work qubits. Full gates and angles in JSON export."
                for r in c["rows"]] + [
                f"GBM {r['levels']} nodes: grid error {r['grid_error_usd']:.4f} USD; {r['per_preparation']['Ry']} Ry and {r['per_preparation']['CNOT']} CNOT per A; peak {r['peak_qubits']} qubits."
                for r in c["grid_sensitivity_gbm"]],
            "limitations": c["limitations"], "comparison": c,
            "reproduce": "python src/studio.py --out docs/study.json"}
