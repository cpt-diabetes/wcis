import pandas as pd
import numpy as np
import plotly.graph_objects as go


def main():
	rawdata = load_data()
	preds = load_preds("2024-03-10-04-06-33-calibrated")

	selected_hospitals = [
		"210009",
		"210029",
	]

	steps_out = 20
	
	for h in selected_hospitals:
		plot_preds(preds, rawdata, h, steps_out)

def load_data():
	rawdata = pd.read_csv("hhs_data_cleaned.csv")
	rawdata.date = pd.to_datetime(rawdata.date)
	rawdata.hospital_id = rawdata.hospital_id.astype(str)
	rawdata.sort_values(["hospital_id", "date"], inplace=True)
	rawdata.reset_index(drop=True, inplace=True)
	return rawdata

def load_preds(pred_id):
	preds = pd.read_csv(f"preds-{pred_id}.csv")
	preds.date = pd.to_datetime(preds.date)
	preds.split_date = pd.to_datetime(preds.split_date)
	preds.hospital_id = preds.hospital_id.astype(str)
	preds.sort_values(["hospital_id", "split_date", "steps_out"], inplace=True)
	preds.reset_index(drop=True, inplace=True)
	return preds

def compute_deltas(data, hospital_id, steps_out):
	data = data[data.hospital_id == hospital_id]

	peak_ind = data.census_covid.idxmax()
	delta_max_row = data.loc[peak_ind]
	delta_max = delta_max_row.census_covid + 0.5 * (delta_max_row.capacity - delta_max_row.census)
	delta_maxs = delta_max - data.census_covid.values

	delta = data.census_covid.diff().clip(0, None).quantile(0.95)
	delta = delta * steps_out
	deltas = np.minimum(delta, delta_maxs)

	return deltas

def get_hospital_name(hospital_id):
	metadata = pd.read_csv("hhs_hospital_meta.csv", usecols=["hospital_id", "hospitalname"])
	metadata = metadata[metadata.hospital_id == hospital_id]
	return metadata.hospitalname.values[0]

def plot_preds(preds, data, hospital_id, steps_out):
	selected_preds = preds[(preds.hospital_id == hospital_id) & (preds.steps_out == steps_out)]
	selected_data = data[(data.hospital_id == hospital_id) & (data.date >= pd.to_datetime("2021-07-01")) & (data.date <= pd.to_datetime("2022-07-01"))]

	deltas = compute_deltas(selected_data, hospital_id, steps_out)

	quantile_cols = selected_preds.filter(like="census_covid_", axis=1).columns.tolist()
	quantiles = [float(q.split("_")[-1]) for q in quantile_cols]
	pred_vals = selected_preds[quantile_cols].values

	fig = go.Figure()

	k = len(quantiles)//2
	for i in range(k):
		j = len(quantiles) - i - 1
		p = round((1 - (2 * quantiles[i])) * 100)
		if p > 90:
			continue
		pred_dates = selected_preds.date.tolist()
		lb, ub = pred_vals[:,i].tolist(), pred_vals[:,j].tolist()
		fig.add_trace(go.Scatter(
			name=f"Preiction ({p}% PI)",
			x=pred_dates + pred_dates[::-1],
			y=ub + lb[::-1],
			mode="lines",
			line=dict(width=0),
			fillcolor="rgba(27, 98, 164, 0.15)",
			hoverinfo="skip",
			fill="toself",
			showlegend=False,
		))

	fig.add_trace(go.Scatter(
		x=selected_preds.date,
		y=pred_vals[:,k],
		name="Prediction",
		mode="lines",
		line=dict(color="rgb(31, 119, 180)"),
	))

	fig.add_trace(go.Scatter(
		x=selected_data.date,
		y=selected_data["census_covid"],
		name="Truth",
		mode="lines",
		line=dict(color="rgb(255, 20, 20)", width=3),
	))

	fig.add_trace(go.Scatter(
		x=selected_data.date,
		y=selected_data["census_covid"].values + deltas,
		name="Delta UB",
		mode="lines",
		line=dict(color="gray", dash="dash"),
		showlegend=False,
	))
	fig.add_trace(go.Scatter(
		x=selected_data.date,
		y=selected_data["census_covid"].values - deltas,
		name="Delta LB",
		mode="lines",
		line=dict(color="gray", dash="dash"),
		showlegend=False,
	))

	h_name = get_hospital_name(hospital_id)
	ub = (selected_data["census_covid"].values + deltas).max() * 1.1

	fig.update_yaxes(title_text="COVID Patients", rangemode="tozero", tickformat=",.0f", range=(0, ub))
	fig.update_layout(title_text=f"Predicted vs Actual COVID Occupancy at {h_name} - {steps_out} Days Out", title_x=0.5, title_y=0.85, title_xanchor="center", title_yanchor="top", title_font_size=15)

	fig.show()


if __name__ == "__main__":
	main()
