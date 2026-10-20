"""Multi-asset derivative risk, variance reduction and an explicit amplitude-estimation oracle."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, xlogy
from scipy.stats import qmc
try:
    from .common import finalize, metric
except ImportError:
    from common import finalize, metric


def option_value(spot, strike, volatility, time, rate=.03, kind="call"):
    spot=np.asarray(spot); time=max(float(time),1e-12)
    d1=(np.log(spot/strike)+(rate+.5*volatility**2)*time)/(volatility*np.sqrt(time));d2=d1-volatility*np.sqrt(time)
    call=spot*ndtr(d1)-strike*np.exp(-rate*time)*ndtr(d2)
    return call if kind=="call" else call-spot+strike*np.exp(-rate*time)


def calibrate_prices(path):
    """Optional user CSV: date followed by positive closing-price columns. No network or secrets."""
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.reader(f))
    names=rows[0][1:]; dates=[row[0] for row in rows[1:]]
    prices=np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    if len(prices)<60 or prices.shape[1]!=len(names) or np.any(prices<=0) or not np.isfinite(prices).all():
        raise ValueError("CSV requires at least 60 complete dates of positive prices")
    if dates!=sorted(set(dates)): raise ValueError("Dates must be unique and in ascending ISO order")
    returns=np.diff(np.log(prices),axis=0); covariance=np.cov(returns,rowvar=False)*252
    covariance=np.atleast_2d(covariance); covariance=.9*covariance+.1*np.diag(np.diag(covariance))
    volatility=np.sqrt(np.diag(covariance));correlation=covariance/np.outer(volatility,volatility)
    return {"assets":names,"spot":prices[-1].tolist(),"volatility":volatility.tolist(),"correlation":correlation.tolist(),"rows":len(prices)}


def portfolio_samples(paths=4096,seed=1729,method="sobol",horizon=21/252,stress=1):
    n=4; spot=np.array([100.,80.,120.,60.]);vol=np.array([.22,.28,.18,.35])*stress
    units=np.array([50.,-30.,40.,20.]);strikes=np.array([100.,85.,115.,65.]);kinds=["call","put","call","put"]
    correlation=.45**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);factor=np.linalg.cholesky(correlation)
    if method=="sobol":
        u=qmc.Sobol(n,scramble=True,seed=seed).random_base2(int(np.log2(paths)))
        if len(u)!=paths:raise ValueError("Sobol path budget must be a power of two")
        z=ndtri(np.clip(u,1e-12,1-1e-12))
    elif method=="antithetic":
        if paths%2:raise ValueError("Antithetic path budget must be even")
        base=np.random.default_rng(seed).standard_normal((paths//2,n));z=np.concatenate([base,-base])
    else:z=np.random.default_rng(seed).standard_normal((paths,n))
    terminal=spot*np.exp((.03-.5*vol**2)*horizon+np.sqrt(horizon)*(z@factor.T)*vol)
    initial=sum(units[j]*option_value(spot[j],strikes[j],vol[j],.5,kind=kinds[j]) for j in range(n))
    future=sum(units[j]*option_value(terminal[:,j],strikes[j],vol[j],.5-horizon,kind=kinds[j]) for j in range(n))
    losses=float(initial)-np.exp(-.03*horizon)*future
    return losses,np.maximum(future,0),float(initial)


def risk_summary(losses,exposure,confidence=.99):
    threshold=float(np.quantile(losses,confidence));tail=losses[losses>=threshold]
    return {"var":threshold,"es":float(tail.mean()),"pfe":float(np.quantile(exposure,confidence)),"mean_loss":float(np.mean(losses))}


def amplitude_oracle(probabilities,payoff):
    """Householder preparation of scenario register + payoff ancilla; exact CPU matrices."""
    probabilities=np.asarray(probabilities,float);payoff=np.asarray(payoff,float)
    if len(probabilities)&(len(probabilities)-1) or len(probabilities)!=len(payoff):raise ValueError("Power-of-two scenario register required")
    if not np.isclose(probabilities.sum(),1) or np.any(probabilities<0) or np.any((payoff<0)|(payoff>1)):raise ValueError("Invalid encoding")
    state=np.zeros(2*len(payoff));state[0::2]=np.sqrt(probabilities*(1-payoff));state[1::2]=np.sqrt(probabilities*payoff)
    e0=np.eye(len(state))[0];v=e0-state
    A=np.eye(len(state)) if np.linalg.norm(v)<1e-14 else np.eye(len(state))-2*np.outer(v,v)/(v@v)
    S0=np.eye(len(state));S0[0,0]=-1;Sgood=np.diag(np.tile([1.,-1.],len(payoff)))
    Q=-A@S0@A.T@Sgood
    assert np.allclose(A.T@A,np.eye(len(state)),atol=1e-12)
    assert np.allclose(Q.T@Q,np.eye(len(state)),atol=1e-12)
    return state,Q


def quantum_estimate(state,Q,powers=(0,1,2,4,8),shots=256,seed=1729):
    rng=np.random.default_rng(seed);counts=[];query_count=0
    for power in powers:
        evolved=np.linalg.matrix_power(Q,power)@state;p=float(np.sum(evolved[1::2]**2))
        success=int(rng.binomial(shots,np.clip(p,0,1)));counts.append((power,success,shots));query_count+=shots*(2*power+1)
    theta=np.linspace(1e-6,np.pi/2-1e-6,32769);log_likelihood=np.zeros_like(theta)
    for power,success,total in counts:
        p=np.clip(np.sin((2*power+1)*theta)**2,1e-12,1-1e-12)
        log_likelihood+=xlogy(success,p)+xlogy(total-success,1-p)
    best=int(np.argmax(log_likelihood));accepted=log_likelihood>=log_likelihood[best]-1.920729
    # A profile-likelihood set may be disconnected; its bounding interval is reported honestly.
    bounds=np.sin(theta[accepted])**2
    return {"estimate":float(np.sin(theta[best])**2),"interval":[float(bounds.min()),float(bounds.max())],
            "queries":query_count,"counts":[{"grover_power":p,"successes":s,"shots":n} for p,s,n in counts]}


def experiment():
    cases=[]
    for stress,title in [(1,"Base market"),(1.8,"Volatility shock")]:
        losses,exposure,pv=portfolio_samples(32768,stress=stress)
        summary=risk_summary(losses,exposure)
        edges=np.linspace(np.quantile(losses,.002),np.quantile(losses,.998),55);hist,_=np.histogram(losses,edges)
        x=((edges[:-1]+edges[1:])/2).tolist()
        convergence=[]
        reference=portfolio_samples(131072,seed=499,stress=stress)[0].mean()
        budgets=[256,1024,4096,16384]
        for method in ["plain","antithetic","sobol"]:
            errors=[]
            for budget in budgets:
                estimates=[portfolio_samples(budget,seed=100+j,method=method,stress=stress)[0].mean() for j in range(12)]
                errors.append(float(np.sqrt(np.mean((np.array(estimates)-reference)**2))))
            convergence.append({"name":method,"values":errors})
        assert summary['es']>=summary['var']
        cases.append({"id":f"risk-{stress}","label":title,"kicker":"MULTI-ASSET DERIVATIVE RISK","title":"The portfolio, under pressure.",
          "description":"Four correlated option positions. Revalue the full book at a 21-day horizon and compare three simulation designs at matched path budgets.",
          "metrics":[metric("99% value at risk",summary['var'],"USD"),metric("99% expected shortfall",summary['es'],"USD"),metric("99% potential exposure",summary['pfe'],"USD"),metric("Scenarios",len(losses),"paths")],
          "charts":[{"title":"Discounted mark-to-market loss distribution","x":x,"xlabel":"Loss / USD","series":[{"name":"Scenario count","values":hist.tolist()}]},
                    {"title":"Mean-loss estimation error · 12 independent runs","x":budgets,"xlabel":"Payoff paths","series":convergence}],
          "table":{"columns":["Asset","Position","Instrument","Spot","Strike","Annual volatility"],"rows":[["A",50,"Call",100,100,.22*stress],["B",-30,"Put",80,85,.28*stress],["C",40,"Call",120,115,.18*stress],["D",20,"Put",60,65,.35*stress]]},
          "method":["Correlated GBM terminal simulation; Black–Scholes revaluation of remaining six-month option maturities. Positions and currency scale are explicit.","Randomized Sobol and antithetic designs are compared with independent Monte Carlo using 12 separate seeds at four budgets. The high-budget independent reference is itself a simulation estimate.","VaR is an empirical loss quantile; expected shortfall averages observations in the empirical tail. PFE uses positive future book value, not losses."],
          "limitations":["Illustrative synthetic asset parameters, not calibrated market data or an investment recommendation.","Lognormal dynamics omit stochastic volatility, defaults, liquidity and funding; tail estimates remain model dependent.","The chart compares mean-loss error. It does not establish quantum advantage or tail-quantile confidence intervals."],
          "reproduce":"python src/studio.py --out docs/study.json"})
    p=np.array([.05,.1,.15,.2,.2,.15,.1,.05]);payoff=np.array([0,.02,.05,.1,.2,.4,.7,1.]);state,Q=amplitude_oracle(p,payoff);truth=float(p@payoff)
    powers=[0,1,2,3,4,5,6,7,8];actual=[float(np.sum((np.linalg.matrix_power(Q,k)@state)[1::2]**2)) for k in powers];analytic=np.sin((2*np.array(powers)+1)*np.arcsin(np.sqrt(truth)))**2
    assert np.allclose(actual,analytic,atol=1e-10)
    estimate=quantum_estimate(state,Q);queries=estimate['queries'];classical=np.random.default_rng(431).binomial(1,truth,queries).mean()
    cases.append({"id":"amplitude","label":"Quantum oracle","kicker":"AMPLITUDE ESTIMATION / EXPLICIT STATEVECTOR","title":"Look inside the quantum estimator.",
      "description":"An eight-scenario register and a payoff ancilla, explicit unitary preparation, amplitude amplification and maximum-likelihood estimation.",
      "metrics":[metric("Exact encoded expectation",truth),metric("Estimated expectation",estimate['estimate']),metric("Oracle preparation calls",queries),metric("Logical qubits",4)],
      "charts":[{"title":"Amplified success probability","x":powers,"xlabel":"Grover power","series":[{"name":"Explicit unitary evolution","values":actual},{"name":"Analytic prediction","values":analytic.tolist()}]}],
      "table":{"columns":["Grover power","Successful shots","Total shots"],"rows":[[r['grover_power'],r['successes'],r['shots']] for r in estimate['counts']]},
      "method":["Prepare amplitudes sqrt(p_j(1-f_j)) and sqrt(p_j f_j) on the scenario/ancilla basis. Householder state preparation is unitary and explicitly inverted inside the Grover operator.",f"A Bernoulli likelihood over Grover powers estimates the bounded expectation. The profile-likelihood bounding interval is {estimate['interval'][0]:.5f} to {estimate['interval'][1]:.5f}; it is an asymptotic, potentially disconnected likelihood set.",f"At the same {queries} preparation-query budget a recorded direct Bernoulli estimate is {classical:.5f}. One random trial is not a comparative performance study."],
      "limitations":["Dense unitary simulation is exponential and is not a compiled fault-tolerant implementation.","This bounded discrete payoff is separate from the portfolio's unbounded continuous loss; no claim that the oracle directly estimates its VaR is made.","State preparation, inverse preparation and Grover queries are counted. Device gates, errors, compilation and classical inference costs are not hidden in a speedup claim."],"reproduce":"python src/studio.py --out docs/study.json"})
    return {"schema_version":1,"slug":"quantum-estimation-lab","title":"Quantum Risk Observatory","subtitle":"From a correlated derivatives book to the quantum estimation oracle.","accent":"#65d6b1","seed":1729,"cases":cases,
      "references":[{"title":"Original portfolio PFE project","url":"https://github.com/FyodorAmanov1/QStockPrediction"},{"title":"Amplitude amplification and estimation","url":"https://arxiv.org/abs/quant-ph/0005055"}],
      "provenance":"New implementation inspired by the application scope of QStockPrediction. Synthetic market parameters; exact CPU statevector experiments. No cloud hardware or API calls."}


if __name__=='__main__':finalize(experiment())
