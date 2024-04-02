import pandas as pd
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import datetime as dt


def evaluate(pred_fn):
    preds = pd.read_csv(pred_fn)
    preds.hospital_id = preds.hospital_id.astype(str)
    preds.sort_values(["hospital_id", "split_date", "steps_out"], inplace=True)

    cfg = {
        "hospital_ids": [str(h) for h in preds.hospital_id.unique().tolist()],
        "split_dates": preds.split_date.unique().tolist(),
        "target_col": "census_covid",
        "forecast_steps": preds.steps_out.max(),
        "days_per_step": 1,
    }
    cfg = SimpleNamespace(**cfg)

    quantile_cols = preds.filter(like="census_covid_", axis=1).columns.tolist()
    cfg.quantiles = [float(q.split("_")[-1]) for q in quantile_cols]

    rawdata = load_data(cfg)
    deltas_by_hosp, delta_limits, max_dict = compute_deltas(cfg, rawdata)

    results = []
    full_results = []
    for split_date in cfg.split_dates:
        for h in cfg.hospital_ids:
            h_preds = preds[(preds.hospital_id == h) & (preds.split_date == split_date)]
            pred_dates = h_preds.date.to_list()
            h_targets = rawdata[(rawdata.hospital_id == h) & (rawdata.date.isin(pd.to_datetime(pred_dates)))]
            deltas = deltas_by_hosp[h]
            df_out = h_preds.copy(deep=True)
            df_out["date"] = pd.to_datetime(df_out["date"])
            df_out.sort_values(by="steps_out", ascending=True, inplace=True)
            df_out["delta"] = deltas
            df_out = df_out.merge(h_targets.loc[:, ['hospital_id', 'date', 'capacity', 'census', 'census_icu',
                                                    'census_covid', 'census_covid_icu', 'admissions_covid']],
                                  on=["hospital_id", "date"], how="left")

            h_preds = h_preds[quantile_cols].to_numpy()
            h_targets = h_targets[cfg.target_col].to_numpy()

            errs = compute_errors(h_preds, h_targets, deltas, cfg.quantiles)
            errs_by_steps_out = compute_errors_steps(h_preds, h_targets, deltas, cfg.quantiles)

            df_out = df_out.merge(errs_by_steps_out, on="steps_out", how="left")
            full_results.append(df_out)
            results.append({"errors": errs, "errors_by_steps_out": errs_by_steps_out})

    full_df = pd.concat(full_results, ignore_index=True)
    err_df = pd.DataFrame([r["errors"] for r in results])
    mean_errs = err_df.mean()

    err_dfs = [r["errors_by_steps_out"] for r in results]
    err_df = pd.concat(err_dfs)
    err_df = err_df.groupby("steps_out").mean()
    err_df = err_df.rename(columns={"ae": "mae", "ape": "mape", "se": "mse"})
    dummy = 1
    print("Mean Errors:")
    print(mean_errs)
    print()

    print("Errors by Steps Out:")
    print(err_df)

    return full_df, results, deltas_by_hosp, delta_limits


def load_data(cfg):
    rawdata = pd.read_csv(Path("preds-01", "hhs_data_cleaned.csv"))
    rawdata.hospital_id = rawdata.hospital_id.astype(str)

    rawdata = rawdata[rawdata.hospital_id.isin(cfg.hospital_ids)].reset_index(drop=True)

    rawdata.date = pd.to_datetime(rawdata.date)
    data_start_date = rawdata.date.min()
    rawdata["date_idx"] = (rawdata.date - data_start_date).dt.days
    rawdata["time_step"] = rawdata["date_idx"] // cfg.days_per_step

    rawdata = rawdata.groupby(["hospital_id", "time_step"]).agg({
        "date": "min",
        "date_idx": "min",
        "capacity": "max",
        "census": "mean",
        "census_icu": "mean",
        "census_covid": "mean",
        "census_covid_icu": "mean",
        "admissions_covid": "sum",
    }).reset_index()

    rawdata.sort_values(["hospital_id", "time_step"], inplace=True)
    rawdata.reset_index(drop=True, inplace=True)

    return rawdata


def wcis(preds, targets, deltas, quantiles):
    k = len(quantiles) // 2
    alphas = quantiles[:k]
    alphas = [2 * q for q in alphas]

    score = cae(preds[:, k], targets, deltas)

    for (i, alpha) in enumerate(alphas):
        j = len(quantiles) - i - 1
        a, b = preds[:, i], preds[:, j]
        score += cis(a, b, targets, deltas, alpha)

    score = score * (1 / (k + 1))

    return score


def cis(l, u, ys, deltas, alpha):
    width_score = (alpha / (2 * deltas)) * (u - l)

    miss_score = ((l > ys) * cae(l, ys, deltas)) + ((u < ys) * cae(u, ys, deltas))

    score = width_score + miss_score
    score = np.minimum(score, 1)
    return score


def cae(preds, truths, deltas):
    f = np.abs(preds - truths) / deltas
    f = np.minimum(f, 1)
    return f


def positive_diff_quantile(s, quant=0.5):
    """
	:param s: series
	:param quant: quantile level
	:return: quant-level quantile of positive diffs of s
	"""
    s_pos = s.diff()
    s_pos = s_pos[s_pos > 0]
    return s_pos.quantile(quant)


def clean_census_for_delta(s_):
    """
	sets census equal to census_covid if census < census_covid
	sets capacity = max(census, census_covid) if capacity is less than either of them
	:param s_: row of rawdata
	:return: "corrected" s_
	"""
    s = s_.copy(deep=True)
    if s["census"] < s["census_covid"]:
        s["census"] = s["census_covid"]
    if s["capacity"] < s["census"]:
        s["capacity"] = s["census"]
    return s


def compute_deltas(cfg, rawdata_):  # is covid census always less than census?
    rawdata = rawdata_.copy(deep=True).apply(lambda row: clean_census_for_delta(row), axis=1)

    peak_inds_ = rawdata.groupby("hospital_id")["census_covid"].idxmax().values
    peak_vals = rawdata.loc[peak_inds_, ["hospital_id", "census_covid", "census"]].copy(deep=True).set_index(
        "hospital_id")

    peak_inds_cap = rawdata.groupby("hospital_id")["capacity"].idxmax().values
    peak_vals_cap = rawdata.loc[peak_inds_cap, ["hospital_id", "capacity"]].copy(deep=True).set_index("hospital_id")
    max_df = pd.concat([peak_vals, peak_vals_cap], axis=1)
    max_df["makes_sense"] = (max_df.census_covid <= max_df.census) & (max_df.census <= max_df.capacity)
    max_df["delta_max"] = max_df.census_covid + 0.5 * ((max_df.capacity - max_df.census).apply(lambda x: max(0, x)))
    max_df["breathing_room"] = max_df.delta_max - max_df.census_covid
    max_df["breathing_room_percentage"] = (max_df.breathing_room / max_df.census_covid) * 100
    delta_limits = max_df["delta_max"].to_dict()

    max_dict = {}
    for hosp, hosp_df_ in rawdata_.groupby("hospital_id"):
        hosp_df = hosp_df_.sort_values(by="date_idx", ascending=True).set_index("date_idx").copy(deep=True)

        horiz_list = [hosp_df["census_covid"].diff(horizon).quantile(0.95) for horizon in
                      range(1, 22)]

        horiz_list_updated = [horiz_list[0]]
        for i in range(1, len(horiz_list)):
            horiz_list_updated.append(max(max(horiz_list_updated[:i]), horiz_list[i]))
        max_dict[hosp] = horiz_list_updated

    deltas_by_stepsout = np.arange(1, cfg.forecast_steps + 1)

    deltas_by_hosp = {}
    for h, hdf in rawdata.groupby("hospital_id"):
        a = hdf[cfg.target_col].diff().abs().to_list()
        aa = [i for i in a if i > 0]
        deltas_by_hosp[h] = np.mean(aa)

    for h in deltas_by_hosp.keys():
        deltas_by_hosp[h] = deltas_by_hosp[h] * deltas_by_stepsout

    return deltas_by_hosp, delta_limits, max_dict


def wis(preds, targets, quantiles):
    k = len(quantiles) // 2
    alphas = quantiles[:k]
    alphas = [2 * q for q in alphas]

    score = 0.5 * np.abs(preds[:, k] - targets)

    for (i, alpha) in enumerate(alphas):
        j = len(quantiles) - i - 1
        a, b = preds[:, i], preds[:, j]
        score += (alpha / 2) * interval_score(a, b, targets, alpha)

    score = score * (1 / (k + 0.5))
    return score


def interval_score(l, u, ys, alpha):
    width_score = u - l

    miss_score = (l - ys) * (ys < l) + (ys - u) * (ys > u)
    miss_score = miss_score * (2 / alpha)

    score = width_score + miss_score
    return score


def mae(preds, targets):
    return np.abs(preds - targets).mean()


def mape(preds, targets):
    return np.nanmean(np.abs(preds - targets) / targets)


def smape(preds, targets):
    return np.nanmean(np.abs(preds - targets) / (np.abs(preds) + np.abs(targets)))


def mse(preds, targets):
    return ((preds - targets) ** 2).mean()


def compute_errors(preds, targets, deltas, quantiles):
    preds_median = preds[:, quantiles.index(0.5)]
    errors = {
        "wcis": wcis(preds, targets, deltas, quantiles).mean(),
        "wis": wis(preds, targets, quantiles).mean(),
        "mae": mae(preds_median, targets),
        # "mape": mape(preds_median, targets),
        # "smape": smape(preds_median, targets),
        "mse": mse(preds_median, targets),
    }
    return errors


def compute_errors_steps(preds, targets, deltas, quantiles):
    preds_median = preds[:, quantiles.index(0.5)]
    err_df = pd.DataFrame({
        "steps_out": np.arange(1, len(targets) + 1),
        "wcis": wcis(preds, targets, deltas, quantiles),
        "wis": wis(preds, targets, quantiles),
        "ae": np.abs(preds_median - targets),
        # "ape": np.abs(preds_median - targets) / targets,
        # "sape": np.abs(preds_median - targets) / ((np.abs(preds_median) + np.abs(targets)) / 2),
        "se": (preds_median - targets) ** 2,
    })
    return err_df


if __name__ == "__main__":
    pred_fn = Path("preds-01", "preds-2024-03-10-04-06-33-calibrated.csv")
    fr, r, dbh, dl = evaluate(pred_fn)

    fr.to_csv(f"facility_results_{dt.datetime.now().strftime('%Y-%b-%d-%H%M')}.csv")
