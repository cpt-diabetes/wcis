using CSV
using DataFrames
using Dates
using Statistics
using Loess
using ProgressMeter


function clean_data()
	println("Loading data...")
	rawdata = DataFrame(CSV.File(
		"rawdata/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility_20240212.csv",
		missingstring=["", "-999999"],
		dateformat="yyyy/mm/dd",
	))

	data = select(
		rawdata,
		"hospital_pk" => "hospital_id",
		"collection_week" => "week",

		"all_adult_hospital_inpatient_beds_7_day_sum" => "capacity",
		"all_adult_hospital_inpatient_bed_occupied_7_day_sum" => "census",

		"total_staffed_adult_icu_beds_7_day_sum" => "capacity_icu",
		"staffed_adult_icu_bed_occupancy_7_day_sum" => "census_icu",

		"total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_sum" => "census_covid",
		"staffed_icu_adult_patients_confirmed_and_suspected_covid_7_day_sum" => "census_covid_icu",

		"previous_day_admission_adult_covid_confirmed_7_day_sum" => "admissions_covid_confirmed",
		"previous_day_admission_adult_covid_suspected_7_day_sum" => "admissions_covid_suspected",

		# "previous_day_total_ED_visits_7_day_sum" => "edvisits",
		# "previous_day_covid_ED_visits_7_day_sum" => "edvisits_covid",

		# "total_patients_hospitalized_confirmed_influenza_7_day_sum" => "census_flu",
		# "icu_patients_confirmed_influenza_7_day_sum" => "census_flu_icu",
		# "previous_day_admission_influenza_confirmed_7_day_sum" => "admissions_flu",
	)
	sort!(data, [:hospital_id, :week])

	filter!(r -> r.week >= Date(2020, 8, 1), data)
	filter!(r -> r.week <= Date(2024, 1, 27), data)

	cols = setdiff(names(data), ["hospital_id", "week"])
	hospital_ids = sort(unique(data.hospital_id))
	weeks = sort(unique(data.week))
	start_week, end_week = extrema(weeks)
	days = start_week:Day(1):(end_week+Day(6))

	hospital_to_state = select(rawdata, :hospital_pk => :hospital_id, :state)
	unique!(hospital_to_state)
	hospital_to_state = Dict(zip(hospital_to_state.hospital_id, hospital_to_state.state))

	println("Loading state data...")
	statedata, states = get_state_data(days, cols)

	println("Disaggregating data...")
	data_dict = Dict((r.hospital_id, r.week) => r for r in eachrow(data))
	cleaned = Array{Union{Missing, Int64}}(missing, length(hospital_ids), length(days), length(cols))
	@showprogress for (i, hospital_id) in enumerate(hospital_ids), (j, week) in enumerate(weeks), (k, col) in enumerate(cols)
		row = get(data_dict, (hospital_id, week), nothing)
		if isnothing(row) continue end

		value = row[col]
		if ismissing(value) continue end

		start_day_index = searchsortedfirst(days, week)
		end_day_index = start_day_index + 6

		state = hospital_to_state[hospital_id]
		state_idx = findfirst(states .== state)

		if isnothing(state_idx)
			trend = ones(7)
		else
			trend = statedata[state_idx, start_day_index:end_day_index, k]
			trend = coalesce.(trend, 0)
		end

		s = sum(trend)
		if s == 0
			trend = ones(7) / 7
		else
			trend = trend / sum(trend)
		end

		vals = value * trend
		vals = round.(Int, vals)

		cleaned[i, start_day_index:end_day_index, k] = vals
	end

	good_hospitals_mask = mean(ismissing.(cleaned), dims=(2,3))[:] .< 0.3
	cleaned = cleaned[good_hospitals_mask, :, :]
	hospital_ids = hospital_ids[good_hospitals_mask]

	println("Imputing missing data...")
	@showprogress for i in 1:size(cleaned, 1), j in 1:size(cleaned, 3)
		fill_ends!(@view cleaned[i, :, j])
		impute_loess!(@view cleaned[i, :, j])
	end

	println("Building dataframe...")
	cleaned_df = DataFrame(
		hospital_id = repeat(hospital_ids, outer=length(cols)*length(days)),
		date = repeat(days, inner=length(hospital_ids), outer=length(cols)),
		col = repeat(cols, inner=length(hospital_ids)*length(days)),
		value = vec(cleaned),
	)
	cleaned_df = unstack(cleaned_df, :col, :value)
	sort!(cleaned_df, [:hospital_id, :date])

	cleaned_df.admissions_covid = cleaned_df.admissions_covid_confirmed .+ cleaned_df.admissions_covid_suspected
	select!(cleaned_df, Not([:admissions_covid_confirmed, :admissions_covid_suspected]))

	println("Writing to file...")
	cleaned_df |> CSV.write("cleaned.csv")

	println("Done!")
	return cleaned_df
end


function impute_loess!(ys::AbstractVector{Union{Missing, Int64}})
	m = ismissing.(ys)
	if !any(m) return end

	m_start = max(1, findfirst(m) - 21)
	m_end = min(length(ys), findlast(m) + 21)
	ys = @view ys[m_start:m_end]

	badmask = ismissing.(ys)
	goodmask = .!badmask

	xs = collect(1:length(ys))

	us = xs[goodmask]
	vs = ys[goodmask]
	vs = convert(Vector{Int64}, vs)

	model = loess(us, vs, span=0.02)

	qs = xs[badmask]
	zs = predict(model, qs)
	zs = round.(Int64, zs)

	ys[badmask] .= zs

	return
end

function fill_ends!(ys::AbstractVector{Union{Missing, Int64}})
	vals_start = findfirst(!ismissing, ys)
	vals_end = findlast(!ismissing, ys)

	if isnothing(vals_start)
		fill!(ys, 0)
		return
	end

	if vals_start > 1
		first_val = ys[vals_start]
		prefix = @view ys[1:vals_start-1]
		fill!(prefix, first_val)
	end

	if vals_end < length(ys)
		last_val = ys[vals_end]
		suffix = @view ys[vals_end+1:end]
		fill!(suffix, last_val)
	end

	return
end

function get_state_data(days, cols)
	statedata = DataFrame(CSV.File(
		"rawdata/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries_20240212.csv",
		missingstring=["", "-999999"],
		dateformat="mm/dd/yyyy HH:MM:SS pp",
	))
	statedata.date = Date.(statedata.date)
	select!(
		statedata,
		"state",
		"date",

		"inpatient_beds" => "capacity",
		"inpatient_beds_used" => "census",

		"total_staffed_adult_icu_beds" => "capacity_icu",
		"staffed_adult_icu_bed_occupancy" => "census_icu",

		"total_adult_patients_hospitalized_confirmed_and_suspected_covid" => "census_covid",
		"staffed_icu_adult_patients_confirmed_and_suspected_covid" => "census_covid_icu",

		"previous_day_admission_adult_covid_confirmed" => "admissions_covid_confirmed",
		"previous_day_admission_adult_covid_suspected" => "admissions_covid_suspected",

		# "edvisits"
		# "edvisits_covid"

		# "total_patients_hospitalized_confirmed_influenza" => "census_flu",
		# "icu_patients_confirmed_influenza" => "census_flu_icu",
		# "previous_day_admission_influenza_confirmed" => "admissions_flu",
	)
	sort!(statedata, [:state, :date])

	states = sort(unique(statedata.state))

	statedata_dict = Dict((r.state, r.date) => r for r in eachrow(statedata))
	statedata_array = Array{Union{Missing, Int64}}(missing, length(states), length(days), length(cols))
	for (i, state) in enumerate(states), (j, day) in enumerate(days)
		row = get(statedata_dict, (state, day), nothing)
		if isnothing(row) continue end
		for (k, col) in enumerate(cols)
			statedata_array[i, j, k] = row[col]
		end
	end

	return statedata_array, states
end


if abspath(PROGRAM_FILE) == @__FILE__
	clean_data()
end
