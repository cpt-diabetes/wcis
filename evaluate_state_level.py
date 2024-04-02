import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import scoring_functions
from pathlib import Path
state_lookup_dc = scoring_functions.state_lookup_dc
WIS = scoring_functions.WIS
WCIS = scoring_functions.WCIS_row_revised


#%% inputs

start_date = "2021-05-09"
end_date = "2022-05-09"

# export_processed_data = True
export_processed_data = False

forecast_folder = "data"
truth_file = Path("data", "truth_incident_hospitalizations_downloaded_29_Feb_2024.csv")

models_to_include = [
    "COVIDhub-baseline",
    "COVIDhub-4_week_ensemble",
]

# states to include (FIPS codes)
reference_locations = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22',
    '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41',
    '42', '44', '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56',
    ]

#%% begin operations

# read in the truth data and format
truth = pd.read_csv(truth_file, usecols=range(1, 5))
truth["dt"] = pd.to_datetime(truth["date"])
truth.sort_values(by=["location", "dt"], inplace=True)
truth.query("location != 'US'", inplace=True)

# sum to weekly
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=8)
truth_concat = []
for loc, df in truth.groupby("location"):
    df["target"] = df["value"].rolling(window=indexer, min_periods=8).sum()
    df["target"] = df["target"] - df["value"]
    df["horizon"] = 1
    df.drop(columns=["date", "value"], inplace=True)
    truth_concat.append(df)
    for horizon in [2, 3, 4]:
        df_ = df.copy(deep=True)
        df_["dt"] = df_["dt"] - pd.to_timedelta(horizon-1, unit="w")
        df_["horizon"] = horizon
        truth_concat.append(df_)
    # add horizon = 1 column, then shift dates to get horizons 2-4
weekly_truth = pd.concat(truth_concat, ignore_index=True)
weekly_truth.rename(columns={"dt": "forecast_date"}, inplace=True)

# import forecast data
concat_list = []
for model in models_to_include:
    print(model)
    submissions = [i for i in os.listdir(forecast_folder+"/"+model) if (i[0] not in ["_", "."] and ".csv" in i)]
    submissions = [i for i in submissions if (pd.to_datetime(i[:10]) >= pd.to_datetime(start_date)
                                              and pd.to_datetime(i[:10]) <= pd.to_datetime(end_date))]
    hosp_counter_list = []
    for submission in tqdm(submissions[::-1]):
        # query for state-level point predictions only, assuming that they will also have quantiles
        submitted_df = pd.read_csv(Path(forecast_folder, model, submission), low_memory=False,
                                   ).query("location != 'US'")
        dummy = 1
        hosp_idx = submitted_df["target"].apply(lambda x: "hosp" in x)
        submitted_df = submitted_df.loc[hosp_idx, :]
        submitted_df["model"] = model
        concat_list.append(submitted_df)

hosp_df = pd.concat(concat_list, ignore_index=True)

# clean up prediction dates to make sure they match
hosp_df["forecast_date"] = pd.to_datetime(hosp_df["forecast_date"])
hosp_df["weekday"] = hosp_df["forecast_date"].dt.weekday
hosp_df["date_shift"] = (hosp_df["weekday"] == 6).astype(int)
hosp_df["forecast_date"] = hosp_df["forecast_date"] + pd.to_timedelta(hosp_df["date_shift"], unit="d")

# process forecast to weekly
hosp_df["target_int"] = hosp_df["target"].apply(lambda x: x.split(" ")[0]).astype(int)
hosp_df["quantile"] = hosp_df["quantile"].fillna(1)

week1 = hosp_df.query("0 < target_int < 8").groupby(
    ["forecast_date", "location", "model", "quantile"],
    as_index=False).sum().loc[:, ["forecast_date", "location", "model", "quantile", "value"]].copy(deep=True)
week1["horizon"] = 1

week2 = hosp_df.query("8 <= target_int < 15").groupby(
    ["forecast_date", "location", "model", "quantile"],
    as_index=False).sum().loc[:, ["forecast_date", "location", "model", "quantile", "value"]].copy(deep=True)
week2["horizon"] = 2

week3 = hosp_df.query("15 <= target_int < 22").groupby(
    ["forecast_date", "location", "model", "quantile"],
    as_index=False).sum().loc[:, ["forecast_date", "location", "model", "quantile", "value"]].copy(deep=True)
week3["horizon"] = 3
week4 = hosp_df.query("22 <= target_int < 29").groupby(
    ["forecast_date", "location", "model", "quantile"],
    as_index=False).sum().loc[:, ["forecast_date", "location", "model", "quantile", "value"]].copy(deep=True)
week4["horizon"] = 4

weekly_df = pd.concat([week1, week2, week3, week4], ignore_index=True)

# format so all quantiles and point pred are in a single row
w_concat_list = []
for t, df in weekly_df.groupby(["forecast_date", "location", "model", "horizon"]):
    quant_dict = df.set_index("quantile")["value"].to_dict()
    alpha_intervals = {}
    for q2 in [i for i in quant_dict.keys() if i < 0.5]:
        alpha = q2 * 2
        l = quant_dict[q2]
        u = quant_dict[1 - q2]
        alpha_intervals[alpha] = (l, u)

    point_prediction = quant_dict[1.0]
    del quant_dict[1.0]
    out_series = pd.Series(
        {"forecast_date": t[0], "location": t[1], "model": t[2], "horizon": t[3],
         "point_prediction": point_prediction,
         "alpha_intervals": alpha_intervals, "quantiles": quant_dict}
    )
    w_concat_list.append(out_series)

forecast_data_df = pd.DataFrame(w_concat_list)
data_df = forecast_data_df.merge(weekly_truth, on=["location", "forecast_date", "horizon"], how="left")

hosp_data = pd.read_csv(
    Path("preds-01", "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries_20240212.csv"))

hosp_data.sort_values(by=["state", "date"], inplace=True)

h_states = hosp_data["state"].unique()
mdf = pd.DataFrame([h_states, [state_lookup_dc(i)[2] for i in h_states]]).transpose()
mdf.columns = ["state", "FIPS"]

hosp_data = mdf.merge(hosp_data, how="right", on="state").copy()
hosp_data["P_date"] = pd.to_datetime(hosp_data["date"])

hosp_data["hospital_onset_covid"] = hosp_data["hospital_onset_covid"].apply(
    lambda x: x.replace(",", "") if type(x) == str else x)

for col in hosp_data.columns:
    if col not in ["FIPS", "P_date", "state", "date"]:
        hosp_data[col] = hosp_data[col].apply(
            lambda x: pd.to_numeric(x.replace(",", "")) if type(x) == str else pd.to_numeric(x))


data_df.query("location in @reference_locations", inplace=True)
hosp_data.query("FIPS in @reference_locations", inplace=True)

#%% hosp capacity change stats

h_df = hosp_data.loc[:, ["FIPS", "P_date", "inpatient_beds"]]

#%% generate delta

delta_tracker = {}
delta_tracker_q = {}
delta_tracker_range = {}
for fips in hosp_data["FIPS"].unique():
    df = h_df.query("FIPS==@fips").sort_values(by="P_date")
    changes = {}
    changes_q = {}
    changes_range = {}
    for horizon in [1, 2, 3, 4]:
        diffs_list = []
        for diff in range(2, horizon*7+1):
            diffs_list = diffs_list + [np.abs(i) for i in df["inpatient_beds"].diff(diff).to_list()
                                       if all([pd.notna(i), np.abs(i) > 0])
                                       # if pd.notna(i)
                                       ]
        changes[horizon] = diffs_list
        changes_q[horizon] = {q: np.quantile(diffs_list, q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        changes_range[horizon] = {p: max(diffs_list)*p for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    delta_tracker[fips] = changes
    delta_tracker_q[fips] = changes_q
    delta_tracker_range[fips] = changes_range

delta_lookup = {}
for name, dd in [("quantile", delta_tracker_q), ("range", delta_tracker_range)]:
    d_concat_list = []
    for fips, h_dict in dd.items():
        for h, q_dict in h_dict.items():
            df = pd.Series(q_dict).reset_index()
            df.columns = ["level", "delta"]
            df["horizon"] = h
            df["location"] = fips
            d_concat_list.append(df)

    delta_df = pd.concat(d_concat_list, ignore_index=True)
    delta_lookup[name] = delta_df

#%% add delta to data_df
# kind = "quantile"
# level = 0.9
processed_data = {}
for kind in ["quantile", "range"]:
    for level in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        merge_df = delta_lookup[kind].query("level==@level").drop(columns=["level"]).copy(deep=True)
        test_df = data_df.merge(merge_df, on=["location", "horizon"], how="left")

        # test_df = test_df.query("model != 'USC-SI_kJalpha'")

        test_df["WIS"] = test_df.apply(WIS, axis=1)
        test_df["WCIS"] = test_df.apply(WCIS, axis=1)
        test_df["delta_type"] = [(kind, level)]*len(test_df)

        baseline_df = test_df.query("model == 'COVIDhub-baseline'").loc[:, ["forecast_date",
                                                                            "location",
                                                                            "horizon",
                                                                            "WIS"]].copy(deep=True)
        baseline_df.rename(columns={"WIS": "baseline_WIS"}, inplace=True)
        test_df = test_df.merge(baseline_df, on=["forecast_date", "location", "horizon"], how="left")
        test_df["relative_WIS"] = test_df["WIS"]/test_df["baseline_WIS"]
        processed_data[(kind, np.round(level, decimals=2))] = test_df.copy(deep=True)


if export_processed_data:
    outfile = open("pre_generated", "wb")
    pickle.dump(processed_data, outfile)
    outfile.close()


