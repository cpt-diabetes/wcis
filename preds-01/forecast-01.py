import pandas as pd
import numpy as np

import darts
from darts import TimeSeries

from darts.models import (
	ARIMA,
	AutoARIMA,
	ExponentialSmoothing,
	Prophet,
	LightGBMModel,
	XGBModel,
	DLinearModel,
	TiDEModel,
	CatBoostModel,
	LinearRegressionModel,
	RandomForest,
)
from darts.utils.likelihood_models import QuantileRegression

from darts.metrics import mae, mape, smape, mse, rmse

from darts.utils.timeseries_generation import datetime_attribute_timeseries, holidays_timeseries, linear_timeseries
from darts import concatenate

from types import SimpleNamespace
from datetime import datetime

import plotly.graph_objects as go
import matplotlib.pyplot as plt


def get_config():
	config = {
		"days_per_step": 1,
		"history_steps": 60,
		"forecast_steps": 21,

		"target_col": "census_covid",
		"cov_cols": ["admissions_covid"],

		"quantiles":  [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99],

		"date_range": pd.date_range("2021-07-01", "2022-07-01", freq="1W-MON"),
		"hospital_ids": get_selected_hospitals(),

		# "date_range": [pd.to_datetime("2021-07-01")],
		# "hospital_ids": ["210009"],

		"model_id": "dlinear",

		"training_mode": "single",

		"run_id": get_run_id(),
	}
	config = SimpleNamespace(**config)
	return config

def main():
	cfg = get_config()
	rawdata = load_data(cfg)

	results = []
	for split_date in cfg.date_range:
		if cfg.training_mode == "single":
			r = [train_single(cfg, rawdata, split_date, h) for h in cfg.hospital_ids]
		elif cfg.training_mode == "multiple":
			r = train_multiple(cfg, rawdata, split_date)
		else:
			raise ValueError(f"Unknown training mode: {cfg.training_mode}")
		results.extend(r)

	save_preds(cfg, results)
	print(f"Completed run: {cfg.run_id}")


### Data ###

def get_selected_hospitals():
	metadata = pd.read_csv("hhs_hospital_meta.csv")
	selected_hospital_meta = metadata[(metadata.city == "Baltimore") & (metadata.hospital_type == "shortterm") & ~metadata.hospital.isin(["BALTIMORE CONVENTION CENTER ALTERNATE CARE SITE", "UMD REHABILITATION &  ORTHOPAEDIC INSTITUTE", "LEVINDALE HEBREW GERIATRIC CENTER AND HOSPITAL", "GRACE MEDICAL CENTER, INC"])]
	selected_hospital_ids = sorted(selected_hospital_meta.hospital_id.values)
	return selected_hospital_ids

def get_hospital_names():
	metadata = pd.read_csv("hhs_hospital_meta.csv")
	hospital_names = metadata.set_index("hospital_id").to_dict()["hospitalname"]
	return hospital_names

def load_data(cfg):
	rawdata = pd.read_csv("hhs_data_cleaned.csv")
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

def load_external_covariates():
	# covs_df = pd.read_csv("covariates.csv")
	# covs_df = covs_df[covs_df.date.isin(rawdata.date)]
	# selected_covs = ["cases", "temperature", "variant_idx", "search_cold_and_flu"]
	# covs = [TimeSeries.from_dataframe(covs_df, time_col="date", value_cols=selected_covs)]
	# covs = covs + [TimeSeries.from_dataframe(rawdata, time_col="date", value_cols=["admissions_combined"])]
	# covs = concatenate(covs, axis=1)
	pass

def get_data(cfg, rawdata, hospital_id):
	data = TimeSeries.from_dataframe(rawdata[rawdata.hospital_id == hospital_id], time_col="date", value_cols=[cfg.target_col])
	covs = [TimeSeries.from_dataframe(rawdata[rawdata.hospital_id == hospital_id], time_col="date", value_cols=[c]) for c in cfg.cov_cols]

	time_covs = [
		datetime_attribute_timeseries(data, "year", cyclic=False),
		datetime_attribute_timeseries(data, "dayofyear", cyclic=True),
		datetime_attribute_timeseries(data, "dayofweek", cyclic=False),
		linear_timeseries(start=data.start_time(), end=data.end_time(), freq=data.freq_str),
	]
	time_covs = concatenate(time_covs, axis=1)

	return data, covs, time_covs


### Training ###

def train_multiple(cfg, rawdata, split_date):
	_data = [get_data(cfg, rawdata, h) for h in cfg.hospital_ids]
	data, covs, time_covs = zip(*_data)

	data_trainval = [d.drop_after(split_date + pd.Timedelta(days=cfg.forecast_steps*cfg.days_per_step)) for d in data]
	train, val = zip(*[d.split_before(split_date) for d in data_trainval])

	hospital_names = get_hospital_names()
	deltas_by_hosp, delta_limits = compute_deltas(cfg, rawdata)


	model = get_model(cfg)
	model.fit(
		train,
		# past_covariates=covs,
		future_covariates=time_covs,
	)

	preds_all = model.predict(
		cfg.forecast_steps,
		series=train,
		num_samples=500 if not model.likelihood is None else None,
		# past_covariates=covs,
		future_covariates=time_covs,
	)

	results = []
	for i, preds in enumerate(preds_all):
		h = cfg.hospital_ids[i]
		h_name = hospital_names[h]

		preds_arr = preds.all_values(copy=False)
		preds_arr.clip(0, None, out=preds_arr)

		deltas = deltas_by_hosp[h]
		delta_max = delta_limits[h] - rawdata[(rawdata.hospital_id == h) & (rawdata.date == split_date)][cfg.target_col].values[0]
		deltas = np.minimum(deltas, delta_max)

		errs = compute_errors(cfg, preds, val[i], deltas, cfg.quantiles)
		errs_by_steps_out = compute_errors_steps(cfg, preds, val[i], deltas, cfg.quantiles)

		print(f"Hospital: {h_name}, split date: {split_date.date()}")
		print(errs)
		# plot_preds_old(preds)
		# plot_preds(preds, data[i], h_name)

		r = {
			"hospital_id": h,
			"split_date": split_date,
			"errors": errs,
			"errors_by_steps_out": errs_by_steps_out,
			"model": model,
			"preds": preds,
		}
		results.append(r)

	return results

def train_single(cfg, rawdata, split_date, hospital_id):
	data, covs, time_covs = get_data(cfg, rawdata, hospital_id)
	data_trainval = data.drop_after(split_date + pd.Timedelta(days=cfg.forecast_steps*cfg.days_per_step))
	train, val = data_trainval.split_before(split_date)

	h_name = get_hospital_names()[hospital_id]

	model = get_model(cfg)
	model.fit(
		train,
		# past_covariates=covs,
		future_covariates=time_covs,
	)

	preds = model.predict(
		cfg.forecast_steps,
		num_samples=500 if not model.likelihood is None else None,
		# past_covariates=covs,
		future_covariates=time_covs,
	)
	preds_arr = preds.all_values(copy=False)
	preds_arr.clip(0, None, out=preds_arr)

	deltas_by_hosp, delta_limits = compute_deltas(cfg, rawdata)
	deltas = deltas_by_hosp[hospital_id]
	delta_max = delta_limits[hospital_id] - rawdata[(rawdata.hospital_id == hospital_id) & (rawdata.date == split_date)][cfg.target_col].values[0]
	deltas = np.minimum(deltas, delta_max)

	errs = compute_errors(cfg, preds, val, deltas, cfg.quantiles)
	errs_by_steps_out = compute_errors_steps(cfg, preds, val, deltas, cfg.quantiles)

	print(f"Hospital: {h_name}, split date: {split_date.date()}")
	print(errs)
	# plot_preds_old(preds)
	# plot_preds(preds, data[i], h_name)

	r = {
		"hospital_id": hospital_id,
		"split_date": split_date,
		"errors": errs,
		"errors_by_steps_out": errs_by_steps_out,
		# "model": model,
		"preds": preds,
	}
	return r

def get_model(cfg):
	if cfg.model_id == "dlinear":
		return DLinearModel(
			input_chunk_length=cfg.history_steps,
			output_chunk_length=cfg.forecast_steps,
			kernel_size=25,
			random_state=0,
			likelihood=QuantileRegression(quantiles=cfg.quantiles),
			pl_trainer_kwargs={"enable_model_summary": False},
		)
	elif cfg.model_id == "tide":
		return TiDEModel(
			input_chunk_length=cfg.history_steps,
			output_chunk_length=cfg.forecast_steps,
			num_encoder_layers=3,
			num_decoder_layers=3,
			decoder_output_dim=32,
			hidden_size=128,
			temporal_width_past=4,
			temporal_width_future=4,
			temporal_decoder_hidden=32,
			use_layer_norm=False,
			dropout=0.05,
			random_state=0,
			pl_trainer_kwargs={"enable_model_summary": False},
			likelihood=QuantileRegression(quantiles=cfg.quantiles),
		)
	elif cfg.model_id == "xgboost":
		return XGBModel(
			lags=cfg.history_steps,
			# lags_past_covariates=history_steps,
			lags_future_covariates=(cfg.history_steps, 1),
			output_chunk_length=cfg.forecast_steps,
			n_estimators=500,
			# max_depth=8,
			# objective="reg:squarederror",
			# objective="reg:squaredlogerror",
			# objective="reg:absoluteerror",
			# booster="dart",
			# bosster="gblinear",
			reg_lambda=0,
			# alpha=0,
			# eta=0.5,
			likelihood="quantile",
			quantiles=cfg.quantiles,
			random_state=0,
			device="cuda:0",
			n_jobs=16,
		)
	else:
		raise ValueError(f"Unknown model: {cfg.model_id}")

def save_preds(cfg, results):
	preds_dfs = []
	for r in results:
		preds_df = r["preds"]
		preds_df = preds_df.quantiles_df(cfg.quantiles)
		preds_df = preds_df.reset_index()
		preds_df.insert(0, "hospital_id", r["hospital_id"])
		preds_df.insert(1, "split_date", r["split_date"])
		preds_df.insert(2, "steps_out", preds_df.index + 1)
		preds_dfs.append(preds_df)
	preds_df = pd.concat(preds_dfs).reset_index(drop=True)

	preds_df.sort_values(["hospital_id", "split_date", "date"], inplace=True)
	preds_df.reset_index(drop=True, inplace=True)

	preds_df.to_csv(f"preds-{cfg.run_id}.csv", index=False)

def get_run_id():
	run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	return run_id


### Error metrics ###

## WCIS ##

def wcis(preds, targets, deltas, quantiles):
	k = len(quantiles)//2
	alphas = quantiles[:k]
	alphas = [2*q for q in alphas]

	score = cae(preds[:,k], targets, deltas)

	for (i, alpha) in enumerate(alphas):
		j = len(quantiles) - i - 1
		a, b = preds[:,i], preds[:,j]
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

def compute_deltas(cfg, rawdata):
	peak_inds = rawdata.groupby("hospital_id").census_covid.idxmax().values
	delta_maxs_df = rawdata.loc[peak_inds]
	delta_maxs_df["delta_max"] = delta_maxs_df.census_covid + 0.5 * (delta_maxs_df.capacity - delta_maxs_df.census)
	delta_maxs_df = delta_maxs_df[["hospital_id", "delta_max"]].set_index("hospital_id")
	delta_limits = delta_maxs_df.to_dict()["delta_max"]

	# deltas_df = rawdata[["hospital_id", target_col]].copy()
	# for t in range(1, cfg.forecast_steps+1):
	# 	deltas_df[f"delta_{t}"] = deltas_df.groupby("hospital_id")[cfg.target_col].diff(t).fillna(0).abs()
	# deltas_df = deltas_df.groupby("hospital_id").quantile(0.95)
	# deltas_by_hosp = deltas_df.to_dict("index")
	# deltas_by_hosp = {h: np.array([v[f"delta_{t}"] for t in range(1, cfg.forecast_steps+1)]) for h, v in deltas_by_hosp.items()}

	deltas_by_stepsout = np.arange(1, cfg.forecast_steps+1)
	# deltas_by_stepsout = np.minimum(deltas_by_stepsout, cfg.forecast_steps/4 + 0.5*deltas_by_stepsout)

	# deltas_by_hosp = rawdata.groupby("hospital_id")[cfg.target_col].agg(lambda x: x.diff().clip(0, None).quantile(0.95))
	deltas_by_hosp = rawdata.groupby("hospital_id")[cfg.target_col].agg(lambda x: x.diff().abs().quantile(0.95))
	deltas_by_hosp = deltas_by_hosp.to_dict()

	for h in deltas_by_hosp.keys():
		deltas_by_hosp[h] = deltas_by_hosp[h] * deltas_by_stepsout

	return deltas_by_hosp, delta_limits

## WIS ##

def wis(preds, targets, quantiles):
	k = len(quantiles)//2
	alphas = quantiles[:k]
	alphas = [2*q for q in alphas]

	score = 0.5 * np.abs(preds[:,k] - targets)

	for (i, alpha) in enumerate(alphas):
		j = len(quantiles) - i - 1
		a, b = preds[:,i], preds[:,j]
		score += (alpha/2) * interval_score(a, b, targets, alpha)

	score = score * (1 / (k + 0.5))
	return score

def interval_score(l, u, ys, alpha):
	width_score = u - l

	miss_score = (l - ys) * (ys < l) + (ys - u) * (ys > u)
	miss_score = miss_score * (2/alpha)

	score = width_score + miss_score
	return score

## All errors ##

def compute_errors_det(cfg, preds, target):
	assert not preds.is_probabilistic
	errors = {
		"mae": mae(target, preds),
		"mape": mape(target, preds)/100,
		"smape": smape(target, preds)/100,
		"mse": mse(target, preds),
		"rmse": rmse(target, preds),
	}
	return errors

def compute_errors_steps_det(cfg, preds, target):
	assert not preds.is_probabilistic

	split_date = preds.time_index[0] - pd.Timedelta(days=cfg.days_per_step)

	target_df = target.pd_dataframe().rename(columns={cfg.target_col: "target"})
	preds_df = preds.pd_dataframe().rename(columns={cfg.target_col: "pred"})
	err_df = pd.merge(target_df, preds_df, left_index=True, right_index=True)
	err_df["steps_out"] = err_df.index - split_date

	err_df["ae"] = (err_df["pred"] - err_df["target"]).abs()
	err_df["ape"] = err_df["ae"] / err_df["target"]
	err_df["sape"] = err_df["ae"] / ((err_df["pred"].abs() + err_df["target"].abs()) / 2)
	err_df["se"] = err_df["ae"]**2

	err_df.reset_index(drop=True, inplace=True)
	err_df.drop(columns=["target", "pred"], inplace=True)
	return err_df

def compute_errors_prob(cfg, preds, target, deltas, quantiles):
	assert preds.is_probabilistic

	preds_values = preds.quantiles_df(quantiles).values
	target_values = target.values()[:,0]
	preds_median = preds.quantile(0.5)
	preds_median_values = preds_median.values()[:,0]

	errors = {
		"wcis": wcis(preds_values, target_values, deltas, quantiles).mean(),
		"wis": wis(preds_values, target_values, quantiles).mean(),
		"mae": mae(target, preds_median),
		"mape": np.nanmean(np.abs((target_values - preds_median_values) / target_values)),
		"smape": np.nanmean(np.abs((target_values - preds_median_values) / ((np.abs(target_values) + np.abs(preds_median_values)) / 2))),
		"mse": mse(target, preds_median),
		"rmse": rmse(target, preds_median),
	}
	return errors

def compute_errors_steps_prob(cfg, preds, target, deltas, quantiles):
	assert preds.is_probabilistic

	split_date = preds.time_index[0] - pd.Timedelta(days=cfg.days_per_step)

	preds_values = preds.quantiles_df(quantiles).values
	target_values = target.values()[:,0]
	preds_median = preds.quantile(0.5).values()[:,0]

	err_df = pd.DataFrame({
		"steps_out": preds.time_index - split_date,
		"wcis": wcis(preds_values, target_values, deltas, quantiles),
		"wis": wis(preds_values, target_values, quantiles),
		"ae": np.abs(preds_median - target_values),
		"ape": np.abs(preds_median - target_values) / target_values,
		"sape": np.abs(preds_median - target_values) / ((np.abs(preds_median) + np.abs(target_values)) / 2),
		"se": (preds_median - target_values)**2,
	})
	return err_df

def compute_errors(cfg, preds, target, deltas=None, quantiles=None):
	if preds.is_probabilistic:
		return compute_errors_prob(cfg, preds, target, deltas, quantiles)
	else:
		return compute_errors_det(cfg, preds, target)

def compute_errors_steps(cfg, preds, target, deltas=None, quantiles=None):
	if preds.is_probabilistic:
		return compute_errors_steps_prob(cfg, preds, target, deltas, quantiles)
	else:
		return compute_errors_steps_det(cfg, preds, target)


### Plotting ###

def plot_preds_old(cfg, preds, data):
	split_date = preds.time_index[0] - pd.Timedelta(days=cfg.days_per_step)
	data_context = data \
		.drop_before(split_date - pd.Timedelta(days=cfg.history_steps*cfg.days_per_step)) \
		.drop_after(split_date + pd.Timedelta(days=(cfg.forecast_steps+3)*cfg.days_per_step))

	plt.figure(figsize=(12, 4))
	data_context.plot()
	preds.plot(label="forecast")
	plt.axvline(x=split_date-pd.Timedelta(days=cfg.days_per_step), color="r", linestyle="--", label="training cutoff")
	plt.legend()
	plt.ylim(0, None)
	plt.show()

def plot_preds(cfg, preds, data, hospital):
	fig = go.Figure()

	if preds.is_probabilistic:
		pred_time_index = preds.time_index.values
		pred_vals = preds.quantiles_df(cfg.quantiles).values

		k = len(cfg.quantiles)//2
		for i in range(k):
			j = len(cfg.quantiles) - i - 1
			p = round((1 - (2 * cfg.quantiles[i])) * 100)
			if p > 90:
				continue
			fig.add_trace(go.Scatter(
				name=f"Preiction ({p}% PI)",
				x=preds.time_index.values,
				y=pred_vals[:,i],
				mode="lines",
				marker=dict(color="#444"),
				line=dict(width=0),
				showlegend=False,
			))
			fig.add_trace(go.Scatter(
				name=f"Prediction ({p}% PI)",
				x=preds.time_index.values,
				y=pred_vals[:,j],
				marker=dict(color="#444"),
				line=dict(width=0),
				mode="lines",
				fillcolor="rgba(27, 98, 164, 0.10)",
				fill="tonexty",
			))

		fig.add_trace(go.Scatter(
			x=preds.time_index.values,
			y=pred_vals[:,k],
			name="Prediction",
			mode="lines",
			line=dict(color="rgb(31, 119, 180)"),
		))

	else:
		fig.add_trace(go.Scatter(
			x=preds.time_index,
			y=preds.values()[:,0],
			name="Prediction",
			mode="lines",
			line=dict(color="rgb(31, 119, 180)"),
		))

	split_date = preds.time_index[0] - pd.Timedelta(days=cfg.days_per_step)
	ctx_hist_days = 21
	ctx_future_days = 3
	data_context = data \
		.drop_before(split_date - pd.Timedelta(days=ctx_hist_days)) \
		.drop_after(split_date + pd.Timedelta(days=(cfg.forecast_steps*cfg.days_per_step)+ctx_future_days))

	fig.add_trace(go.Scatter(
		x=data_context.time_index,
		y=data_context.values()[:,0],
		name="Truth",
		mode="lines",
		line=dict(color="rgb(255, 20, 20)", width=3),
	))

	fig.add_shape(
		type="line",
		x0=split_date, x1=split_date,
		y0=0, y1=1, yref="paper",
		line=dict(color="gray", width=1, dash="dash"),
	)

	fig.update_yaxes(title_text=cfg.target_col, rangemode="tozero", tickformat=",.0f")
	fig.update_layout(title_text=f"Predicted {cfg.target_col} at {hospital} from {split_date.date()}")

	fig.show()

def plot_data(rawdata):
	fig = go.Figure()
	for hospital_id, hospital_data in rawdata.groupby("hospital_id"):
		fig.add_trace(go.Scatter(x=hospital_data.date, y=hospital_data.census_covid, mode="lines", name=f"{hospital_names[hospital_id]}"))
	fig.update_layout(title_text="COVID Census")
	fig.update_yaxes(title_text="COVID Census")
	fig.show()

	fig = go.Figure()
	for hospital_id, hospital_data in rawdata.groupby("hospital_id"):
		fig.add_trace(go.Scatter(x=hospital_data.date, y=hospital_data.admissions_covid, mode="lines", name=f"{hospital_names[hospital_id]}"))
	fig.update_layout(title_text="COVID Admissions")
	fig.update_yaxes(title_text="COVID Admissions")
	fig.show()

def plot_deltas(cfg, deltas_by_hosp):
	delta_fig = go.Figure()
	for k, v in deltas_by_hosp.items():
		delta_fig.add_trace(go.Scatter(
			x=np.arange(1, cfg.forecast_steps+1),
			y=v,
			name=k,
		))
	delta_fig.show()


if __name__ == "__main__":
	main()
