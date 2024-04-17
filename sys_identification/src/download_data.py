from functools import reduce
from typing import Optional
import polars as pl
import os
from datetime import datetime, timezone, timedelta
import requests
import logging
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
import pytz

logger = logging.getLogger(__name__)

assert pl.__version__.split(".")[0] == "0" and pl.__version__.split(".")[1] == "20", "Please upgrade polars to >0.20"

def _load_param_old_db(param:str, start:datetime, end:datetime) -> Optional[pl.LazyFrame]:
	logger.debug(f"{param=} {start=} {end=}")
	# old data is store in this table until 2023-03-09
	cutoff_time = datetime(2023, 3, 10, tzinfo=timezone.utc)
	if start is not None and start >= cutoff_time:
		return None

	if end is not None and end >= cutoff_time:
		end = None

	username = os.getenv("DB_USER")
	password = os.getenv("DB_PASSWORD")
	CONNECTION = f"postgres://{username}:{password}@phlit-db.postgres.database.azure.com:5432/postgres?sslmode=require"

	#convert the names to the old convention
	# replaced with none if does not exist in old database
	params_replacement = {
		"level": "height",
		"pump3_speed": None,
		"pump3_power": None,
		"pressure": None,
		"pump1_speed": "pump1_rpm",
		"pump4_speed": "pump4_rpm",
		"grid_frequency": None
	}

	param_repl = param
	if param in params_replacement.keys():
		param_repl = params_replacement[param]

	# param does not exist in old database
	if param_repl is None:
		return None
	# add one second to the start time, otherwise the same second with some ms already is included, we want the next full second
	filter_start = f"AND time >= '{(start + timedelta(seconds=1)).isoformat()}'" if start is not None else ""
	filter_end = f"AND time <= '{end.isoformat()}'" if end is not None else ""

	agg_fun = f'avg("{param_repl}") as "{param_repl}"' if "_active" not in param_repl else f'bool_and("{param_repl}") as "{param_repl}"'

	query = f"""
		SELECT time_bucket('1 second', time) as ts, {agg_fun}
		from pump_bornholm
		WHERE (device_id = 'revpi-bornholm' OR device_id = 'revpi_borholm' OR device_id = 'roenne-tysklandsvej') {filter_start} {filter_end}
		group by ts
		ORDER BY ts ASC
		"""
	
	logger.debug(f"{query=}")

	data_raw = pl.read_database_uri(query, CONNECTION).lazy()

	# rename back from temporary variable
	data_raw = data_raw.rename({"ts": "time"})

	logger.debug(f"{data_raw.head()}")

	if param in params_replacement.keys():
		data_raw = data_raw.rename({param_repl: param})

	data_raw = data_raw.sort("time")

	logger.debug(f"{data_raw.head()}")

	return data_raw

def _load_param_new_db(param: str, start:datetime, end:datetime) -> Optional[pl.LazyFrame]:
	logger.debug(f"{param=} {start=} {end=}")

	cutoff_time = datetime(2023, 3, 10, tzinfo=timezone.utc)
	if start is not None and start <= cutoff_time:
		start = None

	if end is not None and end <= cutoff_time:
		return None
	
	# add one second to the start time, otherwise the same second with some ms already is included, we want the next full second
	filter_start = f"AND time >= '{(start + timedelta(seconds=1)).isoformat()}'" if start is not None else ""
	filter_end = f"AND m.time <= '{end.isoformat()}'" if end is not None else ""

	# sql = f"""
	# SELECT time_bucket('1 second', m.time) as time, s.name, avg(m.datapoint) as datapoint
	# FROM Measurement m
	# JOIN Sensor s ON m.sensor_id = s.id
	# JOIN Device d ON s.device_id = d.id
	# JOIN SensorType st ON s.type_id = st.id
	# JOIN Gateway g ON d.gateway_id = g.id
	# WHERE g.id = '2d516a5a-de73-45f4-a3a6-d0299a9df2e0' AND ({params}) {filter} {filter_end}
	# group by time, s.name
	# ORDER BY time ASC
	# """
	
	#simplified query since there is only one gateway adding data anyways
	sql = f"""
	SELECT time_bucket('1 second', m.time) as ts, avg(m.datapoint) as "{param}"
	FROM Measurement m
	JOIN Sensor s ON m.sensor_id = s.id
	WHERE s.name = '{param}' {filter_start} {filter_end}
	group by ts
	ORDER BY ts ASC
	"""
	username = os.getenv("DB_USER")
	password = os.getenv("DB_PASSWORD")
	CONNECTION = f"postgres://{username}:{password}@phlit-db.postgres.database.azure.com:5432/postgres?sslmode=require"

	logger.debug(f"{sql=}")
	data_raw = pl.read_database_uri(sql, CONNECTION).lazy()

	# rename back from temporary variable
	data_raw = data_raw.rename({"ts": "time"})
	logger.debug(f"{data_raw.head()}")

	return data_raw

def _load_param_db(param:str, start:datetime, end:datetime) -> pl.LazyFrame:
	logger.debug("load param from old db")
	old_param = _load_param_old_db(param, start, end)
	logger.debug("load param from new db")
	new_param = _load_param_new_db(param, start, end)

	if old_param is not None and new_param is not None:
		# there is some overlap in the data between the two databases
		old_param = old_param.filter(pl.col("time") < new_param.select(pl.col("time").min()).collect().item())

		merged = pl.concat([old_param, new_param], how="vertical_relaxed")
	elif old_param is not None:
		merged = old_param
	elif new_param is not None:
		merged = new_param
	else:
		logger.debug("no data retrieved")
		merged = None

	return merged

def _load_cache(filename) -> Optional[pl.LazyFrame]:
	dirname = os.path.dirname(__file__)
	cache_file = os.path.join(dirname, "..", "data", filename)
	logger.debug(f"{cache_file=}")
	if os.path.isfile(cache_file):
		logger.debug("load cache")
		cached_param = pl.scan_parquet(cache_file)
	else:
		logger.debug("no cache")
		cached_param = None

	return cached_param

def _load_cache_param(param: str) -> Optional[pl.LazyFrame]:
	filename = f"raw_data_{param}.par"
	cached_param = _load_cache(filename)
	return cached_param

def _load_cache_weather(feature: str) -> Optional[pl.LazyFrame]:
	filename = f"weather_{feature}.par"
	cached_param = _load_cache(filename)
	return cached_param

def _save_cache(data: pl.LazyFrame, filename: str) -> None:
	dirname = os.path.dirname(__file__)
	filename_tmp = f"{filename}.tmp"
	cache_file_tmp = os.path.join(dirname, "..", "data", filename_tmp)

	#write to tmp file to not corrupt file if program crashes
	logger.debug(f"save cache {cache_file_tmp}")
	data.sink_parquet(cache_file_tmp)

	cache_file = os.path.join(dirname, "..", "data", filename)

	# move saved file to actual cache file
	logger.debug(f"move {cache_file_tmp} -> {cache_file}")
	os.replace(cache_file_tmp, cache_file)

def _save_cache_param(data: pl.LazyFrame, param: str) -> None:
	filename = f"raw_data_{param}.par"
	_save_cache(data, filename)

def _save_cache_weather(data: pl.LazyFrame, feature: str) -> None:
	filename = f"weather_{feature}.par"
	_save_cache(data, filename)

def _load_param(param: str, end: datetime, update_data: bool = True) -> Optional[pl.LazyFrame]:
	chunk_size_days = 60

	if update_data:
		logger.info(f"Loading param: {param} until {end}")
	else:
		logger.info(f"loading {param} data from cache")

	logger.debug("try loading cache")
	cache = _load_cache_param(param)
	cache_end = cache.select(pl.col("time").max()).collect().item() if cache is not None else None
	if cache_end is None:
		cache_end = datetime(2022, 11, 28, tzinfo=timezone.utc) # 2022-11-19 is the first day with collected data
	logger.debug(f"cache end: {cache_end}")

	if update_data == False:
		logger.info(f"loading from cache done. {param} data until {cache_end}")
		return cache
	
	current_end = cache_end
	if end == None:
		end = datetime.now(timezone.utc)

	while current_end <= end:
		current_end += timedelta(days=chunk_size_days)
		current_retr = min(current_end, end + timedelta(minutes=1))
		new_data = _load_param_db(param, cache_end, current_retr)

		if cache is not None and new_data is not None:
			merged = pl.concat([cache, new_data], how="vertical_relaxed")
		elif cache is not None:
			return cache
		elif new_data is not None:
			merged = new_data
		else:
			continue

		merged.sort(by="time")

		if not merged.collect().is_empty():
			logger.debug("save cache")
			# write data to disk and scanning it back as lazyframe to avoid storing it in memory
			_save_cache_param(merged, param)
			cache = _load_cache_param(param)
		else:
			cache = merged

		cache_end = cache.select(pl.col("time").max()).collect().item()

		# no more data after this point for this parameter and query is just really slow, should come up with a better fix
		if param == "pump3_active" and cache_end > datetime(2023, 4, 4, tzinfo=timezone.utc):
			break

	#return only data before the specified end
	cache = cache.filter(pl.col("time") <= end)

	logger.info(f"Loading param: {param} done")
		
	return cache

def _get_weather(start: datetime, end: datetime, parameter: str ="precip_past1min") -> pl.LazyFrame:
	apikey = os.getenv("DMI_API_KEY")
	data = None
	current_date = end
	stations = {"precip_past1min": "05994", "temp_dry": "06190", "wind_dir": "06197", "wind_speed":"06197", "temp_soil": "06197", "cloud_cover": "06190", "radia_glob": "06193", "humidity": "06190"}
	station = stations[parameter]
	while True:
		# https://confluence.govcloud.dk/display/FDAPI/Meteorological+Observation
		params = {
			"api-key": apikey,
			"stationId": station,
			# "bbox" : "14.57072,54.93433,15.28868,55.36277", #keep exact stations now, but the bounding box could be used to get data from all over bornholm
			"parameterId": parameter,
			"datetime": "../" + current_date.isoformat()
			# somehow only limiting the end works, limit starting does not work
		}
		r = requests.get("https://dmigw.govcloud.dk/v2/metObs/collections/observation/items", params=params)
		j = r.json()

		if r.status_code != 200:
			raise Exception(f"retrieving weather data failed: {j}, request url: {r.request.url}")

		times = []
		values = []
		for obs in j["features"]:
			assert obs["properties"]["parameterId"] == parameter
			assert obs["properties"]["stationId"] == station
			times.append(datetime.fromisoformat(obs["properties"]["observed"].replace("Z", "")))
			values.append(obs["properties"]["value"])

		d = pl.DataFrame({"time": times, "value": values}).with_columns(pl.col("time").cast(pl.Datetime).dt.replace_time_zone("UTC"))

		if data is None:
			data = d
		else:
			data = pl.concat([data, d], rechunk=False)

		if start > data.select(pl.col("time").min()).item():
			break

		current_date = data.select(pl.col("time").min()).item()

	data = data.with_columns(pl.col("time").dt.cast_time_unit("ns"))
	data = data.rechunk().sort(by="time").rename({"value": parameter})

	return data.lazy()

def _get_weather_cached(start: datetime, end: datetime, parameter: str ="precip_past1min") -> pl.LazyFrame:
	logger.info(f"Loading weather: {parameter} until {end}")

	logger.debug("try loading cache")
	cache = _load_cache_weather(parameter)
	cache_end = cache.select(pl.col("time").max()).collect().item() if cache is not None else None
	if cache_end is None:
		cache_end = start if start is not None else datetime(2022, 11, 28, tzinfo=timezone.utc) # 2022-11-19 is the first day with collected data
	logger.debug(f"cache end: {cache_end}")
	
	if end == None:
		end = datetime.now(timezone.utc)

	new_data = _get_weather(start=cache_end, end=end, parameter=parameter)

	if cache is not None and new_data is not None:
		merged = pl.concat([cache, new_data], how="vertical_relaxed")
	elif cache is not None:
		return cache
	else:
		merged = new_data

	merged.sort(by="time")

	if not merged.collect().is_empty():
		logger.debug("save cache")
		# write data to disk and scanning it back as lazyframe to avoid storing it in memory
		_save_cache_weather(merged, parameter)
		cache = _load_cache_weather(parameter)
	else:
		cache = merged

	logger.info(f"Loading weather: {parameter} done")
		
	return cache

def _weather_and_end(parameter: str, start: datetime, end: datetime):
	d = _get_weather_cached(start, end, parameter)
	last = d.select(pl.col("time").max()).collect().item()
	return d, last

def _add_weather_data(data: pl.LazyFrame, end) -> pl.LazyFrame:
	start = datetime(2022, 11, 28, tzinfo=timezone.utc) # 2022-11-19 is the first day with collected data

	weather_params = ["precip_past1min", "temp_dry", "wind_dir", "wind_speed", "temp_soil", "cloud_cover", "radia_glob", "humidity"]

	# get weather for at least the range
	with ThreadPoolExecutor(max_workers=len(weather_params)) as ex:
		p_map_iter = ex.map(partial(_weather_and_end, start=start, end=end), weather_params)

	p_map_list = list(p_map_iter)
	weather_data = [x[0] for x in p_map_list]
	last_times = [x[1] for x in p_map_list]

	# remove data that is after last recorded weather
	last_weather_both = min(last_times)
	data = data.filter(pl.col("time") < last_weather_both)

	merged_weather = data.sort("time")
	for d in weather_data:
		merged_weather = merged_weather.join_asof(d.sort("time"), on="time", strategy="forward")

	return merged_weather

def _filter_data(data: pl.LazyFrame) -> pl.LazyFrame:
	if "level" in data.columns:
		data_filtered = data.filter((pl.col("level") > 0) & (pl.col("level") < 300))
	else:
		data_filtered = data

	for p in ["outflow", "overflow", "level"]:
		if p in data_filtered.columns:
			data_filtered = data_filtered.drop_nulls(subset=p)

	return data_filtered

def _filter_datetime(data: pl.LazyFrame, start_date, end_date) -> pl.LazyFrame:
    start_date = pytz.utc.localize(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S"))
    end_date = pytz.utc.localize(datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S"))
    data = data.sort("time")
    data = data.filter(pl.col("time").is_between(start_date, end_date))
    return data


def _resample_data(data: pl.LazyFrame, resample_time: str, sensors: list) -> pl.LazyFrame:

	return data.sort("time").group_by_dynamic(index_column="time", every=resample_time, check_sorted=False).agg(pl.col(sensors).mean())


def save_static_copy(data: pl.LazyFrame, path: str):
    model_types = ["outflow_miso", "pump1_power_siso", "pump3_power_siso", "pump4_power_siso", "pressure_miso", "all_features"]

    for model_type in model_types:

        if model_type == "outflow_miso":
            DF_SLICE = ["time","level", "outflow", 'pump1_speed', 'pump4_speed', 'pump3_speed']
        
        elif model_type == "pump1_power_siso":
            DF_SLICE = ["time", "pump1_speed", "pump1_power"]

        elif model_type == "pump3_power_siso":
            DF_SLICE = ["time", "pump3_speed", "pump3_power"]
        
        elif model_type == "pump4_power_siso":
            DF_SLICE = ["time", "pump4_speed", "pump4_power"]
        
        elif model_type == "pressure_miso":
            DF_SLICE = ["time", "pump1_speed", "pump4_speed", "pump3_speed", "outflow", "pressure", "level"]
            
        elif model_type == "all_features":
            DF_SLICE = ["time", "pump1_power", "pump3_power", "pump4_power", 
              "pump1_speed", "pump3_speed", "pump4_speed", 
              "outflow", "pump3_active", "pump2_active", 
              "level", "pressure", "overflow", "inflow", 
              "precip_past1min", "temp_dry", "wind_dir",
              "wind_speed", "temp_soil", "cloud_cover", "radia_glob", "humidity"]

        data[DF_SLICE].write_parquet(path + model_type + ".par")


def _add_inflow(data: pl.LazyFrame) -> pl.LazyFrame:
	from data_prep.kalman_filter import Kalmeistro
	kalman = Kalmeistro(1, 18)

	cache = _load_cache_param("inflow")
	cache_end = cache.select(pl.col("time").max()).collect().item() if cache is not None else None
	if cache_end is None:
		cache_end = datetime(2022, 11, 28, tzinfo=timezone.utc) # 2022-11-19 is the first day with collected data

	data_missing_inflow = data.filter(pl.col("time") > cache_end - timedelta(hours=1))
	needed_columns = data_missing_inflow.select(pl.col("level", "outflow", "overflow", "time")).collect()

	if cache_end < needed_columns.select(pl.col("time").max()).item():
		inflow, *_ = kalman.predict(needed_columns.select(pl.col("level")).to_numpy(), needed_columns.select(pl.col("outflow")).to_numpy(), needed_columns.select(pl.col("overflow")).to_numpy())

		missing_inflow = needed_columns.select(pl.col("time"), inflow=pl.lit(inflow)).lazy()
		missing_inflow = missing_inflow.filter(pl.col("time") > cache_end)

		if cache is not None:
			inflow_data = pl.concat([cache, missing_inflow], how="vertical_relaxed")
		else:
			inflow_data = missing_inflow

		logger.info("save cache")
		# collect().lazy() to fix some operation not available as streaming
		_save_cache_param(inflow_data.collect().lazy(), "inflow")
		inflow_data = _load_cache_param("inflow")
	else:
		inflow_data = cache

	data = data.join(inflow_data, on="time", how="left")

	return data

def get_data(measurement_parameters=None, update_data=False, end=None, add_weather_data=True,
			 filter_data=True, filter_datetime=False, start_date=None, end_date=None,
			 resample=False, resample_time= None) -> pl.LazyFrame:
	

	logger = logging.getLogger(__name__)
	retrieval_start = datetime.now(timezone.utc) - timedelta(minutes=5)

	measurement_parameters = set(measurement_parameters)
	measurement_parameters_retrieval = measurement_parameters.copy()

	#set end to now if no explicit time provided used later to limit since the end of all columns might not be the same
	if end is None:
		end = datetime.now(timezone.utc)

	#add timezone if datetime is naive for comparing
	if end.tzinfo is None or end.tzinfo.utcoffset(end) is None:
		end = end.replace(tzinfo=timezone.utc)

	#check if db vars are set
	if os.getenv("DB_USER") is None:
		raise Exception("No DB_USER environment variable found")
	if os.getenv("DB_PASSWORD") is None:
		raise Exception("No DB_PASSWORD environment variable found")
	
	if "inflow" in measurement_parameters_retrieval:
		add_inflow = True
		measurement_parameters_retrieval.remove("inflow")
		measurement_parameters_retrieval.add("outflow")
		measurement_parameters_retrieval.add("level")
		measurement_parameters_retrieval.add("overflow")
	else:
		add_inflow = False
	
	logger.info("Start loading parameters")
	# use spawn to avoid getting stuck https://docs.pola.rs/user-guide/misc/multiprocessing/
	with ProcessPoolExecutor(max_workers=len(measurement_parameters_retrieval), mp_context=mp.get_context('spawn')) as ex:
		param_data_ex = ex.map(partial(_load_param, end=end, update_data=update_data), measurement_parameters_retrieval, chunksize=1)
		param_data = [p for p in param_data_ex if p is not None]

	logger.info("Start joining parameters")
	data = reduce(lambda x, y: x.join(y, on="time", how="outer_coalesce"), param_data)

	data = data.sort("time")

	data = data.filter(pl.col("time") <= min(end, retrieval_start))

	logger.info("Done joining parameters")
	
	# after this time point only changes in values are reported so forward fill
	logger.debug("interpolate data where only changes are reported")
	data = data.with_columns(pl.when(pl.col("time") > datetime(2024, 2, 7, tzinfo=timezone.utc)).then(pl.exclude("time").forward_fill()).otherwise(pl.exclude("time")))

	if add_inflow:
		logger.info("Start adding inflow")
		data = _filter_data(data)
		data = _add_inflow(data)
		logger.info("Done adding inflow")

	if filter_data:
		data = _filter_data(data)

	if filter_datetime:
		data = _filter_datetime(data, start_date, end_date)

	if resample:
		data = _resample_data(data, resample_time, measurement_parameters)
		
	data = data.select(pl.col("time"), pl.col(measurement_parameters))

	if add_weather_data:
		if os.getenv("DMI_API_KEY") is None:
			raise Exception("No DMI_API_KEY environment variable found")
		logger.info("adding weather data to data")
		data = _add_weather_data(data, end=min(end, retrieval_start))
		logger.info("Done adding weather")

	return data

if __name__ == "__main__":
	FORMAT = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
	
	logging.basicConfig(level=logging.INFO, format=FORMAT)
	
	params = ["pump1_power", "pump3_power", "pump4_power", 
              "pump1_speed", "pump3_speed", "pump4_speed", 
              "outflow", "pump3_active", "pump2_active", 
              "level", "pressure", "overflow", 
              "precip_past1min", "temp_dry", "wind_dir",
              "wind_speed", "temp_soil", "cloud_cover", "radia_glob", "humidity"]
	
	data = get_data(measurement_parameters=params, 
                    add_weather_data=True, 
				    update_data=False,
					resample=True, resample_time="20s",				 
				    filter_datetime=True, start_date="2023-05-01 00:00:00", end_date="2024-02-20 00:00:00")
	
	
	logger.info("done with call")

	data = data.collect()

	save_static_copy(data, path="./data/static_models/")

	logger.info("a static copy of the data has been saved in data_prep/static_models/")
	