import pandas as pd
import numpy as np
import os

class weather_data_pre_processor():

    def __init__(self, time_transformer):
        self.time_transformer = time_transformer

    def extract_attributes(self, path_to_data, attributes, print_out = False):
        weather_data = pd.read_csv(os.path.join(os.getcwd(), path_to_data), sep = ',')
        if print_out:
            print(weather_data.head())
        column_names = weather_data.columns.tolist()
        for attribute in attributes:
            for column in column_names:
                if str(attribute) == str(column):
                    column_names.remove(column)
                    break
        result_data = weather_data.drop(columns=column_names)
        if print_out:
            print(result_data)
        return result_data
    
    def set_time_line(self, data, start, end, freq, drop = []):
        timeline = pd.DataFrame()
        timeline["date"] = pd.date_range(start, end, freq = freq, tz = 'Europe/Berlin')
        timeline['date'] = pd.to_datetime(timeline['date'])
        timeline["Drop"] = 0
        timeline.set_index('date', inplace =True)
        if len(drop) > 0:
            for item in drop:
                rng = pd.date_range(start = pd.to_datetime(item), periods=24, freq="h", tz = 'Europe/Berlin')
                timeline.loc[rng, "Drop"] = 1
        timeline.drop(timeline[timeline['Drop']==1].index, inplace=True)
        timeline.interpolate()
        result = pd.merge(data, timeline, left_index=True, right_index=True, how="inner")
        result.drop(columns = ["Drop"], inplace = True)
        return result

    def create_timepoints(self, data, attributes = [], sample_time = '5min', interpolation_value = "high" ):
        data_new = pd.DataFrame()
        for attribute in attributes:
            attr = data.loc[:,[attribute]]
            attr = attr.resample(sample_time).mean()
            attr = attr.interpolate()
            data_new[attribute] = attr
        return data_new    
        

