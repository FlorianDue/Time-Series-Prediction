import pandas as pd
import numpy as np

class TimeSeries:

    def __init__(self, start, end):
        self.timeseries = pd.DataFrame()
        self.start = start
        self.end = end
    
    def add_daytime(self, sampling_interval, timezone = 'Europe/Berlin', day = None, dayName = None, monthName = True, month = None, 
                    season = None, season_interval = None, holiday = None, daytime = None, dayofyear = None, dayvalues = None, index = 'date'):
        #todo get timezone 0
        self.timeseries["date"] = pd.date_range(self.start, self.end, freq = sampling_interval, tz = timezone) if timezone else pd.date_range(self.start, self.end, freq = sampling_interval)
        if day:
            self.add_day(day)
        if month:
            self.add_month(month)
        if season or season_interval:
            self.add_season(season_interval)
        if holiday:
            self.add_holiday(holiday)
        if daytime:
            self.add_timevalues()
        if dayofyear:
            self.add_day_of_year()
        if dayvalues:
            self.add_dayvalues()
        self.timeseries.set_index(index, inplace = True)

    def add_day_of_year(self):
        self.timeseries["dayOfYear"] = self.timeseries["date"].dt.dayofyear

    def add_day(self, dayName = False):
        self.timeseries["day"] = self.timeseries["date"].dt.day_name()
        if dayName:
            self.timeseries = pd.merge(self.timeseries, pd.get_dummies(self.timeseries['day']).astype(int), left_index=True, right_index=True)
    
    def add_month(self, monthName = False):
        self.timeseries["month"] = pd.DatetimeIndex(self.timeseries["date"]).month
        if monthName:
            self.timeseries = pd.merge(self.timeseries, pd.get_dummies(self.timeseries['month']).astype(int), left_index=True, right_index=True)

    def add_season(self, interval = None):
        self.timeseries["season"] = pd.DatetimeIndex(self.timeseries["date"]).quarter
        if interval != None:
            self.add_seasonName(interval)

    def add_seasonName(self, interval):
        seasons = ["winter", "spring", "summer", "fall"]
        for season in seasons:
            self.timeseries[season] = 0
        for element in interval:
            self.timeseries.set_index("date", inplace = True)
            self.timeseries.loc[str(element["start"]):str(element["end"]), str(element["key"])] = 1
            self.timeseries.reset_index(inplace=True)
    
    def add_holiday(self, holidays):
        self.timeseries.set_index("date", inplace = True)
        holiday = pd.DataFrame()
        holiday["holiday"] = (self.timeseries["day"] == "Sunday").astype(int)
        holiday["weekday"] = (self.timeseries["day"] != "Sunday").astype(int)
        holiday["weekendday"] = (self.timeseries["day"] == ("Sunday" or "Saturday")).astype(int)
        self.timeseries = pd.merge(self.timeseries, holiday , left_index = True, right_index = True)
        for holiday in holidays:
            self.timeseries.loc[holiday[0], holiday[1]] = 1
        self.timeseries.reset_index(inplace=True)
    
    def add_timevalues(self):
        self.timeseries['hour'] = pd.DatetimeIndex(self.timeseries["date"]).hour
        self.timeseries['minutes'] = pd.DatetimeIndex(self.timeseries["date"]).minute
        temp2 = ((self.timeseries['hour']*60) + self.timeseries["minutes"])*(2.*np.pi/1440)
        self.timeseries["hour_sin"] = np.sin(temp2)
        self.timeseries["hour_cos"] = np.cos(temp2)
    
    def add_dayvalues(self):
        temp = 2 * np.pi * self.timeseries['dayOfYear'] / 365.25
        self.timeseries["day_sin"] = np.sin(temp)
        self.timeseries["day_cos"] = np.cos(temp)
    

