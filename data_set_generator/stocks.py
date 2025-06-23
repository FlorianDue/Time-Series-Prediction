import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, maxabs_scale, MinMaxScaler
import uuid


class DataGenerator:

    def __init__(self, timeseries, stocks = []):
        self.stocks = stocks
        self.timeseries = timeseries
        self.test_data_x = pd.DataFrame()
        self.test_data_y = pd.DataFrame()
        self.train_data_x = pd.DataFrame()
        self.train_data_y = pd.DataFrame()
        self.vali_data_x = pd.DataFrame()
        self.vali_data_y = pd.DataFrame()
        self.transformer = None
        self.train_list_x = []
        self.vali_list_x = []
        self.test_list_x = []
        self.train_list_y = []
        self.vali_list_y = []
        self.test_list_y = []


    def sample_data(self, remove_items = ["Open", "High", 'Low', 'Date', "Volume"], resize_interval = {"keys":[[]], "lower":[0], "upper": [1], "strategie":[] }, 
                identifier = False, train = 0.7, vali = 0.15, test = 0.15):
        if identifier == False:
            if len(self.stocks) > 1:
                for stock in self.stocks:
                    stock['uid'] = uuid.uuid4()
        self.transformer = MinMaxScaler(feature_range =(resize_interval['lower'][0], resize_interval['upper'][0]))
        stock_list = []
        for stock in self.stocks:
            stock = stock[~stock.index.duplicated(keep='first')]
            stock = pd.concat([stock, self.timeseries], axis = 1, join='inner').reindex(stock.index)
            #stock = stock.dropna(how='any')
            stock.reset_index(inplace = True)
            if remove_items != None:
                stock.drop(remove_items, axis=1, inplace=True)
            if len(resize_interval["keys"][0]) > 0:
                stock[resize_interval["keys"][0]] = self.transformer.fit_transform(stock[resize_interval["keys"][0]])
            stock_list.append(stock)
        self.stocks = stock_list
        if len(self.stocks) > 1:
            for i in range(len(self.stocks)):
                self.stocks[i].loc[:, 'uid'] = (i+1)/len(self.stocks)

    #todo execute from loop within the single stations
    def data_split(self, training, testing, history, validating = None, keys = ['Close']):
        if validating != None:
            print(self.stocks[0])
            temp = pd.DataFrame()
            #todo use complete list and sort for the timestap to create station independent batches
            for stock in self.stocks:
                stock[keys] = self.transformer.fit_transform(stock[keys])
                targets = stock["Close"]
                targets = targets.shift(-history)
                #remove the shifted elements
                targets = targets[0:len(targets)-history].copy()
                stock = stock[0:len(targets)-history].copy()

                first = round(training*len(stock))
                second = round((training+validating)*len(stock))
                third = len(stock)
                
                

                self.train_list_x.append(stock[0:first])
                self.vali_list_x.append(stock[first-history:second])
                self.test_list_x.append(stock[second-history:third])

                self.train_list_y.append(targets[0:first])
                self.vali_list_y.append(targets[first-history:second])
                self.test_list_y.append(targets[second-history:third])

                #attributs
                self.train_data_x = pd.concat([self.train_data_x, stock[0:first]], axis = 0, ignore_index = False)
                self.vali_data_x = pd.concat([self.vali_data_x, stock[first-history:second]], axis = 0, ignore_index = False)
                self.test_data_x = pd.concat([self.test_data_x, stock[second-history:third]], axis = 0, ignore_index = False)
                #targets
                self.train_data_y = pd.concat([self.train_data_y, targets[0:first]], axis = 0, ignore_index = False)
                self.vali_data_y = pd.concat([self.vali_data_y, targets[first-history:second]], axis = 0, ignore_index = False)
                self.test_data_y = pd.concat([self.test_data_y, targets[second-history:third]], axis = 0, ignore_index = False)

            





        




