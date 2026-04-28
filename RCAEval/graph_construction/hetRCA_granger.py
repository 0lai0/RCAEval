import numpy as np
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def _make_stationary(series, max_diff=2, adf_threshold=0.05):
    """Difference series until ADF passes, up to max_diff."""
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    n_diff = 0

    while n_diff < max_diff and len(s) >= 5:
        try:
            _, p_val, *_ = adfuller(s, autolag="AIC")
        except Exception:
            break
        if p_val < adf_threshold:
            break
        s = np.diff(s)
        n_diff += 1
    return s


def granger(
    data,
    maxlag=None,
    p_val_threshold=0.05,
    test=None,
    apply_fdr=False,
    fdr_method="fdr_bh",
    preprocess_stationarity=False,
    return_stats=False,
):
    assert test in [None, "ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]

    if maxlag is None:
        maxlag = 3

    node_names = data.columns.to_list()
    n_nodes = len(node_names)
    adj = np.zeros((n_nodes, n_nodes))

    series_map = {}
    for name in node_names:
        vals = data[name].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if preprocess_stationarity:
            vals = _make_stationary(vals)
        series_map[name] = vals

    tested_pairs = []
    raw_pvals = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            y = series_map[node_names[i]]
            x = series_map[node_names[j]]
            if len(y) < 5 or len(x) < 5:
                continue
            n = min(len(y), len(x))
            if n < 5:
                continue

            pair_data = np.column_stack([y[-n:], x[-n:]])
            try:
                output = grangercausalitytests(pair_data, maxlag, verbose=False)
            except Exception:
                continue

            lag_pvals = []
            for _, out in output.items():
                stats_dict = out[0]
                if test is None:
                    lag_pvals.extend([v[1] for v in stats_dict.values()])
                elif test in stats_dict:
                    lag_pvals.append(stats_dict[test][1])

            if not lag_pvals:
                continue
            tested_pairs.append((i, j))
            raw_pvals.append(float(np.min(lag_pvals)))

    n_before = int(sum(p < p_val_threshold for p in raw_pvals))
    n_after = 0
    corrected = []

    if raw_pvals:
        if apply_fdr:
            reject, corrected, _, _ = multipletests(
                raw_pvals, alpha=p_val_threshold, method=fdr_method
            )
            keep_flags = reject
        else:
            corrected = raw_pvals
            keep_flags = [p < p_val_threshold for p in raw_pvals]

        for (i, j), keep in zip(tested_pairs, keep_flags):
            if keep:
                adj[i, j] = 1
                n_after += 1

    stats = {
        "n_pairs": len(tested_pairs),
        "n_before_correction": n_before,
        "n_after_correction": n_after,
        "apply_fdr": bool(apply_fdr),
        "fdr_method": fdr_method if apply_fdr else None,
        "preprocess_stationarity": bool(preprocess_stationarity),
        "p_val_threshold": float(p_val_threshold),
        "corrected_pvals": corrected,
    }

    if return_stats:
        return adj, stats
    return adj
