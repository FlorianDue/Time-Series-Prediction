import pandas as pd
import numpy as np
import os

class Bikes:

    def __init__(self, path_to_bikes):
        self.path_to_bikes = path_to_bikes
        self.bikes = pd.DataFrame() 
        self.station_list = []
        self.load_bikes()
        

    def load_bikes(self):
        self.bikes = pd.read_csv(os.path.join(os.getcwd(), self.path_to_bikes))
        self.bikes.rename(columns={'Datum': 'date', 'Fahrräder': 'bikes', 'Station_frei': 'station'}, inplace=True)
        self.bikes.drop(columns=["Station_fest"])
        self.bikes.set_index("date", inplace=True)

    def set_classes(self, highest_class):
        self.bikes.loc[self.bikes['bikes'] > highest_class, 'bikes'] = highest_class

    def separate_stations(self, key = 'uid'):
        self.station_list = [g.copy() for _, g in self.bikes.groupby('uid')]
        
        
        