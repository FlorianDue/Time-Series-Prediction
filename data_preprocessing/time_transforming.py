import pandas as pd

class TimeTransformer:

    def __init__(self):
        pass

    def timestamp_to_time(self, data, key, unit, set_as_idx = False, timezone = 'Europe/Berlin'):
        if set_as_idx:
            data[set_as_idx] = pd.to_datetime(data[key], unit=unit) 
            data.set_index(set_as_idx, inplace =True)
        else:
            data = pd.to_datetime(data, unit=unit)
        if timezone:
            data.index = data.index.tz_localize('UTC') \
                .tz_convert(timezone)
        return data
