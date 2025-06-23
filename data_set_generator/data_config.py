from data_preprocessing.weather import weather_data_pre_processor
from data_preprocessing.time_transforming import TimeTransformer
from data_preprocessing.bikes import Bikes 
from data_preprocessing.timeseries import TimeSeries
from data_set_generator.bike_prediction import DataGenerator

class BikeDataConfig:


    def __init__(self, bikes_path = './/data//bikes.csv', weather_path = './/data//weather.csv', highest_class = 8):
        self.bikes_path = bikes_path
        self.weather_path = weather_path

        self.seasons = [ {"start": "2017-12-31", "end":"2018-02-28", "key":"winter"},
            {"start": "2018-03-01", "end":"2018-05-31", "key":"spring"},
            {"start": "2018-06-01", "end":"2018-08-31", "key":"summer"},
            {"start": "2018-09-01", "end":"2018-11-30", "key":"fall"},
            {"start": "2018-12-01", "end":"2019-02-28", "key":"winter"},
            {"start": "2019-03-01", "end":"2019-05-31", "key":"spring"},
            {"start": "2019-06-01", "end":"2019-08-31", "key":"summer"},
            {"start": "2019-09-01", "end":"2019-10-31", "key":"fall"}
            ] 

        self.holidays = [
            ['2018-01-01','holiday'],
            ['2018-03-30','holiday'],
            ['2018-04-02','holiday'],
            ['2018-05-01','holiday'],
            ['2018-05-10','holiday'],
            ['2018-05-21','holiday'],
            ['2018-05-31','holiday'],
            ['2018-10-03','holiday'],
            ['2018-11-01','holiday'],
            ['2018-12-25','holiday'],
            ['2018-12-26','holiday'],
            ['2019-01-01','holiday'],
            ['2019-04-19','holiday'],
            ['2019-04-22','holiday'],
            ['2019-05-01','holiday'],
            ['2019-05-30','holiday'],
            ['2019-06-10','holiday'],
            ['2019-06-20','holiday'],
            ['2019-10-03','holiday'] 
        ]


        self.transformer = weather_data_pre_processor(TimeTransformer())
        self.data = self.transformer.extract_attributes(self.weather_path, ["dt","temp", "temp_min", "temp_max", "pressure", "humidity", "wind_speed", "clouds_all"])
        self.data = self.transformer.time_transformer.timestamp_to_time(data = self.data, key = "dt", unit = 's', set_as_idx="date")
        self.data = self.transformer.set_time_line(data = self.data, start='01/01/2018 00:00', end='12/31/2019 23:59', freq='h', drop = ['2018-03-25', '2018-10-28', '2019-03-31', '2019-10-29'])
        self.data = self.transformer.create_timepoints(self.data, attributes = ['temp','humidity','wind_speed'])

        self.bikes = Bikes(self.bikes_path)
        self.bikes.set_classes(highest_class = 8)
        self.bikes.separate_stations()

        self.timeseries = TimeSeries(start = '01/01/2018 00:00', end = '12/31/2019 23:59')
        self.timeseries.add_daytime(sampling_interval = '5min', day = True, dayName = True, month = True, monthName = True, season = True, season_interval = self.seasons, holiday = self.holidays, daytime = True)

        self.d_gen = DataGenerator(bikes = self.bikes.bikes, station_list = self.bikes.station_list, weather = self.data,  timeseries = self.timeseries.timeseries)
        self.station_list = None, 
        self.bikes = None
        self.sample_data()

    def sample_data(self):
        self.station_list, self.bikes = self.d_gen.sample_data(remove_items = ["timestamp", "Station_fest", 'day',     'Friday',     'Monday',   'Saturday',     'Sunday',
                'Thursday',    'Tuesday',  'Wednesday', "holiday", 1, 2, 3, 4, 5,6,7,8,9,10,11,12,
                 'weekendday', 'hour', 'minutes', "month", "fall", "spring", "summer", "winter"], resize_interval = {"keys":[["latitude", "longitude" ]], "lower":[-1], "upper": [1], "strategie":[] }, 
                station_key = 'uid', train = 0.7, vali = 0.15, test = 0.15)

