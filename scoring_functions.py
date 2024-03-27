import numpy as np
import pandas as pd


def CAE(x, y, delta):
    score = np.abs(x - y) / delta
    if score < 1:
        return score
    else:
        return 1


def CIS_revised(l, u, y, delta, alpha):
    if delta == 0.0:
        return 1.0
    else:
        width_term = (alpha / (2*delta)) * (u-l)
        if y < l:
            # miss_term = (1 - alpha) * CAE(l, y, delta)
            miss_term = CAE(l, y, delta)
        elif y > u:
            # miss_term = (1 - alpha) * CAE(u, y, delta)
            miss_term = CAE(u, y, delta)
        else:
            miss_term = 0

        return min(1.0, width_term + miss_term)


def WCIS_revised(y, delta, m, intervals, f_cis=CIS_revised, f_cae=CAE):
    """

    expects y, delta, m as scalars, intervals as dict of form {alpha: (l, u)}
    y - truth
    delta - utility threshold
    m - predictive median
    """

    K = len(intervals)

    inner_score = f_cae(m, y, delta)
    for alpha, (l, u) in intervals.items():
        inner_score += f_cis(l, u, y, delta, alpha)
    return (1 / (K + 1)) * inner_score

    # inner_score = 0.5*f_cae(m, y, delta)
    # for alpha, (l, u) in intervals.items():
    #     inner_score += (alpha/2)*f_cis(l, u, y, delta, alpha)
    # return min(1, (1 / (K + 0.5)) * inner_score)


def WCIS_row_revised(row, delta_col="delta", target_col="target", f_wcis=WCIS_revised):
    delta = row[delta_col]
    y = row[target_col]

    # intervals = {}
    # quantiles = {np.float16(k): i for k, i in row["Quantiles"].items()}
    # for i in range(len(quantiles.keys()) // 2):  # iterate through lower interval key indices
    #     low_quantile = sorted(quantiles.keys())[i]
    #     high_quantile = 100.0 - low_quantile
    #     alpha = 1 - ((high_quantile - low_quantile) / 100)
    #     intervals[np.round(alpha, decimals=2)] = (quantiles[low_quantile], quantiles[high_quantile])

    quantiles = row["quantiles"]
    intervals =row["alpha_intervals"]
    m = quantiles[0.5]

    score = f_wcis(y, delta, m, intervals)

    return score


def WCIS_heatmap_revised(
        wcis_table,
        f_wcis_row=WCIS_row_revised,
):
    delta_dict = {
        "cases": "known_weighted_mean",
        "deaths": "known_weighted_mean",
        # "hosp": "available_beds_horizon"
        # "hosp": "available_beds_horizon_scaled"
        "hosp": "available_beds_hist_change"
    }

    concat_list = []
    for epi_, delta_col in delta_dict.items():
        table_slice = wcis_table.query("Epi==@epi_").copy()
        table_slice["WCIS"] = table_slice.apply(f_wcis_row, axis=1, **dict(delta_col=delta_col), )
        concat_list.append(table_slice)

    wcis_table_out = pd.concat(concat_list, ignore_index=True)
    return "WCIS", wcis_table_out


#%%
def WIS(col):
    quantiles = col["quantiles"]
    y = col["target"]
    summation = 0
    for t, q in quantiles.items():
        # t = t_ / 100
        if y <= q:
            indicator = 1
        else:
            indicator = 0

        summation += 2 * (indicator - t) * (q - y)

    # return summation / (2 * len(quantiles) + 1)
    return summation / len(quantiles)

#%%

state_csv = pd.read_csv('/Users/maximilianmarshall/Dropbox (Personal)/Corona/risk_model/DORK_0/data/state_lookup.csv',
                        dtype={2: str}, header=None)
state_csv.rename(columns={2: 'FIPS'}, inplace=True)
state_csv['FIPS'] = state_csv['FIPS'].apply(lambda x: '0'+x if len(x) < 2 else x)

state_csv = pd.concat([state_csv,
                       pd.DataFrame(['District of Columbia', 'DC', '11'],
                                    index=[0, 1, 'FIPS']).transpose()],
                      ignore_index=True)

# def state_lookup(input):
#     """
#     if input = state name, returns fips string
#     if input = fips string, return state name
#     """
#     return state_exchange[input]


def state_lookup(check):
    for i in state_csv.index:
        for entry in state_csv.loc[i, :].values:
            if check == entry:
                return tuple(state_csv.loc[i, :].values)

            # for lowercase state abbreviations
            if type(check) == str:
                if check.upper() == entry:
                    return tuple(state_csv.loc[i, :].values)

    return None, None, None


def state_lookup_dc(check):
    tup = state_lookup(check)
    if tup[0] == "District of Columbia":
        return ("D.C.", tup[1], tup[2])
    elif tup[0] == "Northern Mariana Islands":
        return ("N. Mariana Islands", tup[1], tup[2])
    else:
        return tup
