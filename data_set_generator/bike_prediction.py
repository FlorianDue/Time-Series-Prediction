import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, maxabs_scale


class DataGenerator:

    def __init__(self, bikes, station_list, weather, timeseries, timesshift = 3):
        self.bikes = bikes
        self.weather = weather
        self.timeseries = timeseries
        self.station_list = station_list
        self.test_data_x = pd.DataFrame()
        self.test_data_y = pd.DataFrame()
        self.train_data_x = pd.DataFrame()
        self.train_data_y = pd.DataFrame()
        self.vali_data_x = pd.DataFrame()
        self.vali_data_y = pd.DataFrame()
        self.timesshift = timesshift
        self.transformer = RobustScaler()
        self.train_list_x = []
        self.vali_list_x = []
        self.test_list_x = []
        self.train_list_y = []
        self.vali_list_y = []
        self.test_list_y = []


    def sample_data(self, remove_items = ["timestamp", "Station_fest", 'day',     'Friday',     'Monday',   'Saturday',     'Sunday',
                'Thursday',    'Tuesday',  'Wednesday', "holiday", 1, 2, 3, 4, 5,6,7,8,9,10,11,12,
                 'weekendday', 'hour', 'minutes', "month", "fall", "spring", "summer", "winter"], resize_interval = {"keys":[["latitude", "longitude" ]], "lower":[-1], "upper": [1], "strategie":[] }, 
                station_key = 'uid', train = 0.7, vali = 0.15, test = 0.15):
        dataset =  pd.DataFrame()
        dataset = pd.concat([self.weather, self.timeseries], axis = 1, join='inner').reindex(self.timeseries.index)
        dataset = dataset.dropna(how='any')
        self.bikes = pd.DataFrame()
        for station in self.station_list:
            station.index = pd.to_datetime(station.index, utc=True) \
                    .tz_convert('Europe/Berlin')
            station = station[~station.index.duplicated(keep='first')]
            df = pd.concat([station, dataset], axis = 1, join='inner').reindex(station.index)
            df = df.dropna(how='any')
            if remove_items != None:
                df.drop(remove_items, axis=1, inplace=True)
            df.reset_index(inplace = True)
            self.bikes = pd.concat([self.bikes, df], axis = 0, ignore_index = False)
        for element in resize_interval["keys"][0]:
            self.bikes[element] = (self.bikes[element]+90)/180
        self.station_list = [g.copy() for _, g in self.bikes.groupby('uid')]
        for station in self.station_list:
            station.drop([station_key, 'date'], axis=1, inplace=True)
        self.bikes.drop([station_key], axis=1, inplace=True)
        return self.station_list, self.bikes

    #todo execute from loop within the single stations
    def data_split(self, training, testing, timestep_shift, validating = None, keys = ['bikes', 'season', "temp", "humidity", "wind_speed"]):
        if validating != None:
            temp = pd.DataFrame()
            #todo use complete list and sort for the timestap to create station independent batches
            for station in self.station_list:
                targets = station["bikes"].astype(int)
                targets = targets.shift(-timestep_shift)
                #remove the shifted elements
                targets = targets[0:len(targets)-timestep_shift].copy()
                station = station[0:len(targets)-timestep_shift].copy()

                first = round(training*len(station))
                second = round((training+validating)*len(station))
                third = len(station)
                for key in keys:
                    station[key] = station[key]/station[key].max() 

                self.train_list_x.append(station[0:first])
                self.vali_list_x.append(station[first:second])
                self.test_list_x.append(station[second:third])

                self.train_list_y.append(targets[0:first])
                self.vali_list_y.append(targets[first:second])
                self.test_list_y.append(targets[second:third])

                #attributs
                self.train_data_x = pd.concat([self.train_data_x, station[0:first]], axis = 0, ignore_index = False)
                self.vali_data_x = pd.concat([self.vali_data_x, station[first:second]], axis = 0, ignore_index = False)
                self.test_data_x = pd.concat([self.test_data_x, station[second:third]], axis = 0, ignore_index = False)
                #targets
                self.train_data_y = pd.concat([self.train_data_y, targets[0:first]], axis = 0, ignore_index = False)
                self.vali_data_y = pd.concat([self.vali_data_y, targets[first:second]], axis = 0, ignore_index = False)
                self.test_data_y = pd.concat([self.test_data_y, targets[second:third]], axis = 0, ignore_index = False)

            





        




