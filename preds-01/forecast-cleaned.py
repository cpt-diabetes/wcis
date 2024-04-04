import pandas as pd
import numpy as np

import darts
from darts import TimeSeries
from darts.models import (
	XGBModel,
	DLinearModel,
	TiDEModel,
)
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import datetime_attribute_timeseries, linear_timeseries
from darts import concatenate

from types import SimpleNamespace
from datetime import datetime


def get_config():
	config = {
		"days_per_step": 1,
		"history_steps": 60,
		"forecast_steps": 21,

		"target_col": "census_covid",
		"cov_cols": ["admissions_covid"],

		"quantiles": [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
					  0.8, 0.85, 0.9, 0.95, 0.975, 0.99],

		"date_range": pd.date_range("2021-07-01", "2022-07-01", freq="1W-MON"),
		"hospital_ids": get_selected_hospitals(),

		"model_id": "tide",

		"run_id": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
	}
	config = SimpleNamespace(**config)
	return config

def main():
	cfg = get_config()
	rawdata = load_data(cfg)

	results = []
	for split_date in cfg.date_range:
		for h in cfg.hospital_ids:
			r = train_single(cfg, rawdata, split_date, h)
			results.append(r)

	save_preds(cfg, results)
	print(f"Completed run: {cfg.run_id}")


### Data ###
def get_selected_hospitals():
	metadata = pd.read_csv("hhs_hospital_meta.csv")
	selected_hospital_meta = metadata[(metadata.city == "Baltimore") & (metadata.hospital_type == "shortterm") & ~metadata.hospital.isin(["BALTIMORE CONVENTION CENTER ALTERNATE CARE SITE", "UMD REHABILITATION &  ORTHOPAEDIC INSTITUTE", "LEVINDALE HEBREW GERIATRIC CENTER AND HOSPITAL", "GRACE MEDICAL CENTER, INC"])]
	selected_hospital_ids = sorted(selected_hospital_meta.hospital_id.values)
	return selected_hospital_ids

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

def train_single(cfg, rawdata, split_date, hospital_id):
	data, covs, time_covs = get_data(cfg, rawdata, hospital_id)
	data_trainval = data.drop_after(split_date + pd.Timedelta(days=cfg.forecast_steps*cfg.days_per_step))
	train, val = data_trainval.split_before(split_date)

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

	r = {
		"hospital_id": hospital_id,
		"split_date": split_date,
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


if __name__ == "__main__":
	main()
