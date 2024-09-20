import logging
logging.basicConfig(level=logging.INFO)

import os
import sys
import importlib
import pickle
import itertools
import numpy as np
import pandas as pd

import yfinance as yf

from tabulate import tabulate
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import copy
shiftRange = [0]
stretchRange = [1]
filterRange = [5]
noiseRange = [0]
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
import gym
pd.options.mode.chained_assignment = None
from scipy import signal
import pandas_datareader as pdr
import requests
from io import StringIO

import random
import datetime
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

gamma = 0.4
learningRate = 0.0001
targetNetworkUpdate = 1000
learningUpdatePeriod = 1
capacity = 100000
batchSize = 32
experiencesRequired = 1000
numberOfNeurons = 512
dropout = 0.2
epsilonStart = 1.0
epsilonEnd = 0.01
epsilonDecay = 10000
alpha = 0.1
filterOrder = 5
gradientClipping = 1
rewardClipping = 1
L2Factor = 0.000001
GPUNumber = 0

fictives = {
    'Linear Upward' : 'LINEARUP',
    'Linear Downward' : 'LINEARDOWN',
    'Sinusoidal' : 'SINUSOIDAL',
    'Triangle' : 'TRIANGLE',
}
stocks = {
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ',
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Tesla' : 'TSLA',
    'TeslaAug' : 'TSLA_aug',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}
indices = {
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ'
}
companies = {
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Tesla' : 'TSLA',
    'TeslaAug' : 'TSLA_aug',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}
models = {
    'Buy and Hold' : 'BuyAndHold',
    'Sell and Hold' : 'SellAndHold',
    'Trend Following Moving Averages' : 'MovingAveragesTF',
    'Mean Reversion Moving Averages' : 'MovingAveragesMR'
}
strategiesAI = {
    'TDQN' : 'TDQN'
}
MIN = 100
MAX = 200
PERIOD = 252
saving = False
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')

class tradingStrategy(ABC):
    @abstractmethod
    def chooseAction(self, state):
        pass
    @abstractmethod
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        pass
    @abstractmethod
    def testing(self, testingEnv, trainingEnv, rendering=False, showPerformance=False):
        pass
class BuyAndHold(tradingStrategy):
    def chooseAction(self, state):
        return 1
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        trainingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
        if verbose:
            logging.info("No training is required as the simple Buy and Hold trading strategy does not involve any tunable parameters.")
        if rendering:
            trainingEnv.render()
        if plotTraining:
            logging.info("No training results are available as the simple Buy and Hold trading strategy does not involve any tunable parameters.")
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('B&H')
        return trainingEnv
    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))
        if rendering:
            testingEnv.render()
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('B&H')
        return testingEnv
class SellAndHold(tradingStrategy):
    def chooseAction(self, state):
        return 0
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        trainingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
        if verbose:
            logging.info("No training is required as the simple Sell and Hold trading strategy does not involve any tunable parameters.")
        if rendering:
            trainingEnv.render()
        if plotTraining:
            logging.info("No training results are available as the simple Sell and Hold trading strategy does not involve any tunable parameters.")
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('S&H')
        return trainingEnv
    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))
        if rendering:
            testingEnv.render()
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('S&H')
        return testingEnv
class MovingAveragesTF(tradingStrategy):
    def __init__(self, parameters=[5, 10]):
        self.parameters = parameters
    def setParameters(self, parameters):
        self.parameters = parameters
    def processState(self, state):
        return state[0]
    def chooseAction(self, state):
        state = self.processState(state)
        shortAverage = np.mean(state[-self.parameters[0]:])
        longAverage = np.mean(state[-self.parameters[1]:])
        if(shortAverage >= longAverage):
            return 1
        else:
            return 0
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)
        trainingEnv.reset()
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1
        if verbose:
            iterations = dimension - 1
            length = 0
            while iterations > 0:
                length += iterations
                iterations -= 1
        for shorter in range(bounds[0], bounds[1], step):
            for longer in range(bounds[0], bounds[1], step):
                if(shorter < longer):
                    if(verbose):
                        logging.info("".join(["Training progression: ", str(count), "/", str(length)]), end='\r', flush=True)
                    self.setParameters([shorter, longer])
                    done = 0
                    while done == 0:
                        _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
                    performanceAnalysis = PerformanceEstimator(trainingEnv.data)
                    performance = performanceAnalysis.computeSharpeRatio()
                    results[i][j] = performance
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    trainingEnv.reset()
                    count += 1
                j += 1
            i += 1
            j = 0
        trainingEnv.reset()
        self.setParameters([bestShort, bestLong])
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
        if rendering:
            trainingEnv.render()
        if plotTraining:
            self.plotTraining(results, bounds, step, trainingEnv.marketSymbol)
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('MATF')
        return trainingEnv
    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False):
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))
        if rendering:
            testingEnv.render()
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data,)
            analyser.displayPerformance('MATF')
        return testingEnv
    def plotTraining(self, results, bounds, step, marketSymbol, savePlots=False):
        x = range(bounds[0], bounds[1], step)
        y = range(bounds[0], bounds[1], step)
        xx, yy = np.meshgrid(x, y, sparse=True)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Long Window Duration')
        ax.set_ylabel('Short Window Duration')
        ax.set_zlabel('Sharpe Ratio')
        ax.plot_surface(xx, yy, results, cmap=plt.cm.get_cmap('jet'))
        ax.view_init(45, 45)
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_MATFOptimization3D', '.png']))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111,
                             ylabel='Short Window Duration',
                             xlabel='Long Window Duration')
        graph = ax.imshow(results,
                          cmap='jet',
                          extent=(bounds[0], bounds[1], bounds[1], bounds[0]))
        plt.colorbar(graph)
        plt.gca().invert_yaxis()
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_MATFOptimization2D', '.png']))
class MovingAveragesMR(tradingStrategy):
    def __init__(self, parameters=[5, 10]):
        self.parameters = parameters
    def setParameters(self, parameters):
        self.parameters = parameters
    def processState(self, state):
        return state[0]
    def chooseAction(self, state):
        state = self.processState(state)
        shortAverage = np.mean(state[-self.parameters[0]:])
        longAverage = np.mean(state[-self.parameters[1]:])
        if(shortAverage <= longAverage):
            return 1
        else:
            return 0
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, plotTraining=False, showPerformance=False):
        bounds = trainingParameters[0]
        step = trainingParameters[1]
        dimension = math.ceil((bounds[1] - bounds[0])/step)
        trainingEnv.reset()
        results = np.zeros((dimension, dimension))
        bestShort = 0
        bestLong = 0
        bestPerformance = -100
        i = 0
        j = 0
        count = 1
        if verbose:
            iterations = dimension - 1
            length = 0
            while iterations > 0:
                length += iterations
                iterations -= 1
        for shorter in range(bounds[0], bounds[1], step):
            for longer in range(bounds[0], bounds[1], step):
                if(shorter < longer):
                    if(verbose):
                        logging.info("".join(["Training progression: ", str(count), "/", str(length)]), end='\r', flush=True)
                    self.setParameters([shorter, longer])
                    done = 0
                    while done == 0:
                        _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
                    performanceAnalysis = PerformanceEstimator(trainingEnv.data)
                    performance = performanceAnalysis.computeSharpeRatio()
                    results[i][j] = performance
                    if(performance > bestPerformance):
                        bestShort = shorter
                        bestLong = longer
                        bestPerformance = performance
                    trainingEnv.reset()
                    count += 1
                j += 1
            i += 1
            j = 0
        trainingEnv.reset()
        self.setParameters([bestShort, bestLong])
        done = 0
        while done == 0:
            _, _, done, _ = trainingEnv.step(self.chooseAction(trainingEnv.state))
        if rendering:
            trainingEnv.render()
        if plotTraining:
            self.plotTraining(results, bounds, step, trainingEnv.marketSymbol)
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('MAMR')
        return trainingEnv
    def testing(self, trainingEnv, testingEnv, rendering=False, showPerformance=False, savePlots=False):
        testingEnv.reset()
        done = 0
        while done == 0:
            _, _, done, _ = testingEnv.step(self.chooseAction(testingEnv.state))
        if rendering:
            testingEnv.render()
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('MAMR')
        return testingEnv
    def plotTraining(self, results, bounds, step, marketSymbol, savePlots=False):
        x = range(bounds[0], bounds[1], step)
        y = range(bounds[0], bounds[1], step)
        xx, yy = np.meshgrid(x, y, sparse=True)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Long Window Duration')
        ax.set_ylabel('Short Window Duration')
        ax.set_zlabel('Sharpe Ratio')
        ax.plot_surface(xx, yy, results, cmap=plt.cm.get_cmap('jet'))
        ax.view_init(45, 45)
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_MAMROptimization3D', '.png']))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111,
                             ylabel='Short Window Duration',
                             xlabel='Long Window Duration')
        graph = ax.imshow(results,
                          cmap='jet',
                          extent=(bounds[0], bounds[1], bounds[1], bounds[0]))
        plt.colorbar(graph)
        plt.gca().invert_yaxis()
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_MAMROptimization2D', '.png']))
class TimeSeriesAnalyser:
    def __init__(self, timeSeries):
        self.timeSeries = timeSeries
    def plotTimeSeries(self):
        pd.plotting.register_matplotlib_converters()
        plt.figure(figsize=(10, 4))
        plt.plot(self.timeSeries.index, self.timeSeries.values, color='blue')
        plt.title("Plot Title")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()
    def timeSeriesDecomposition(self, model='multiplicative'):
        decomposition = seasonal_decompose(self.timeSeries, model=model, period=5, extrapolate_trend='freq')
        plt.rcParams.update({'figure.figsize': (16,9)})
        decomposition.plot()
        plt.show()
    def stationarityAnalysis(self):
        logging.info("Stationarity analysis: Augmented Dickey-Fuller test (ADF):")
        results = adfuller(self.timeSeries, autolag='AIC')
        logging.info("ADF statistic: " + str(results[0]))
        logging.info("p-value: " + str(results[1]))
        logging.info('Critial values (the time series is not stationary with X% condifidence):')
        for key, value in results[4].items():
            logging.info(str(key) + ': ' + str(value))
        if results[1] < 0.05:
            logging.info("The ADF test affirms that the time series is stationary.")
        else:
            logging.info("The ADF test could not affirm whether or not the time series is stationary...")
    def cyclicityAnalysis(self):
        plt.rcParams.update({'figure.figsize': (16,9)})
        pd.plotting.autocorrelation_plot(self.timeSeries)
        plt.show()
        _, axes = plt.subplots(2, figsize=(16, 9))
        plot_acf(self.timeSeries, lags=21, ax=axes[0])
        plot_pacf(self.timeSeries, lags=21, ax=axes[1])
        plt.show()
        _, axes = plt.subplots(1, 10, figsize=(17, 9), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()[:10]):
            pd.plotting.lag_plot(self.timeSeries, lag=i+1, ax=ax)
            ax.set_title('Lag ' + str(i+1))
        plt.show()
class DataAugmentation:
    def shiftTimeSeries(self, tradingEnv, shiftMagnitude=0):
        newTradingEnv = copy.deepcopy(tradingEnv)
        if shiftMagnitude < 0:
            minValue = np.min(tradingEnv.data['Volume'])
            shiftMagnitude = max(-minValue, shiftMagnitude)
        newTradingEnv.data['Volume'] += shiftMagnitude
        return newTradingEnv
    def streching(self, tradingEnv, factor=1):
        newTradingEnv = copy.deepcopy(tradingEnv)
        returns = newTradingEnv.data['Close'].pct_change() * factor
        for i in range(1, len(newTradingEnv.data.index)):
            newTradingEnv.data['Close'][i] = newTradingEnv.data['Close'][i-1] * (1 + returns[i])
            newTradingEnv.data['Low'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['Low'][i]/tradingEnv.data['Close'][i]
            newTradingEnv.data['High'][i] = newTradingEnv.data['Close'][i] * tradingEnv.data['High'][i]/tradingEnv.data['Close'][i]
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]
        return newTradingEnv
    def noiseAddition(self, tradingEnv, stdev=1):
        newTradingEnv = copy.deepcopy(tradingEnv)
        for i in range(1, len(newTradingEnv.data.index)):
            price = newTradingEnv.data['Close'][i]
            volume = newTradingEnv.data['Volume'][i]
            priceNoise = np.random.normal(0, stdev*(price/100))
            volumeNoise = np.random.normal(0, stdev*(volume/100))
            newTradingEnv.data['Close'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['Low'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['High'][i] *= (1 + priceNoise/100)
            newTradingEnv.data['Volume'][i] *= (1 + volumeNoise/100)
            newTradingEnv.data['Open'][i] = newTradingEnv.data['Close'][i-1]
        return newTradingEnv
    def lowPassFilter(self, tradingEnv, order=5):
        newTradingEnv = copy.deepcopy(tradingEnv)
        newTradingEnv.data['Close'] = newTradingEnv.data['Close'].rolling(window=order).mean()
        newTradingEnv.data['Low'] = newTradingEnv.data['Low'].rolling(window=order).mean()
        newTradingEnv.data['High'] = newTradingEnv.data['High'].rolling(window=order).mean()
        newTradingEnv.data['Volume'] = newTradingEnv.data['Volume'].rolling(window=order).mean()
        for i in range(order):
            newTradingEnv.data['Close'][i] = tradingEnv.data['Close'][i]
            newTradingEnv.data['Low'][i] = tradingEnv.data['Low'][i]
            newTradingEnv.data['High'][i] = tradingEnv.data['High'][i]
            newTradingEnv.data['Volume'][i] = tradingEnv.data['Volume'][i]
        newTradingEnv.data['Open'] = newTradingEnv.data['Close'].shift(1)
        newTradingEnv.data['Open'][0] = tradingEnv.data['Open'][0]
        return newTradingEnv
    def generate(self, tradingEnv):
        tradingEnvList = []
        for shift in shiftRange:
            tradingEnvShifted = self.shiftTimeSeries(tradingEnv, shift)
            for stretch in stretchRange:
                tradingEnvStretched = self.streching(tradingEnvShifted, stretch)
                for order in filterRange:
                    tradingEnvFiltered = self.lowPassFilter(tradingEnvStretched, order)
                    for noise in noiseRange:
                        tradingEnvList.append(self.noiseAddition(tradingEnvFiltered, noise))
        return tradingEnvList


class YahooFinance:
    def __init__(self):
        self.data = pd.DataFrame()

    def getDailyData(self, marketSymbol, startingDate, endingDate):
        try:
            ticker = yf.Ticker(marketSymbol)
            data = ticker.history(start=startingDate, end=endingDate)
            self.data = self.processDataframe(data)
        except Exception as e:
            logging.error(f"Failed to download {marketSymbol}, from {startingDate} to {endingDate}: {e}")
            self.data = pd.DataFrame()
        return self.data

    def processDataframe(self, dataframe):
        dataframe = dataframe.reset_index()
        dataframe['Date'] = dataframe['Date'].dt.date
        dataframe.set_index('Date', inplace=True)
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]
        return dataframe

class CSVHandler:
    def dataframeToCSV(self, name, dataframe):
        path = name + '.csv'
        dataframe.to_csv(path)

    def CSVToDataframe(self, name):
        path = name + '.csv'
        return pd.read_csv(path,
                           header=0,
                           index_col='Date',
                           parse_dates=True)


class StockGenerator:
    def linearUp (self, startingDate, endingDate, minValue=MIN, maxValue=MAX):
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        linearUpward = pd.DataFrame(index=DowJones.index)
        length = len(linearUpward.index)
        prices = np.linspace(minValue, maxValue, num=length)
        linearUpward['Open'] = prices
        linearUpward['High'] = prices
        linearUpward['Low'] = prices
        linearUpward['Close'] = prices
        linearUpward['Volume'] = 100000
        return linearUpward
    def linearDown (self, startingDate, endingDate, minValue=MIN, maxValue=MAX):
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        linearDownward = pd.DataFrame(index=DowJones.index)
        length = len(linearDownward.index)
        prices = np.linspace(minValue, maxValue, num=length)
        prices = np.flip(prices)
        linearDownward['Open'] = prices
        linearDownward['High'] = prices
        linearDownward['Low'] = prices
        linearDownward['Close'] = prices
        linearDownward['Volume'] = 100000
        return linearDownward
    def sinusoidal(self, startingDate, endingDate, minValue=MIN, maxValue=MAX, period=PERIOD):
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        sinusoidal = pd.DataFrame(index=DowJones.index)
        length = len(sinusoidal.index)
        t = np.linspace(0, length, num=length)
        prices = minValue + maxValue / 2 * (np.sin(2 * np.pi * t / period) + 1) / 2
        sinusoidal['Open'] = prices
        sinusoidal['High'] = prices
        sinusoidal['Low'] = prices
        sinusoidal['Close'] = prices
        sinusoidal['Volume'] = 100000
        return sinusoidal
    def triangle(self, startingDate, endingDate, minValue=MIN, maxValue=MAX, period=PERIOD):
        downloader = YahooFinance()
        DowJones = downloader.getDailyData('DIA', startingDate, endingDate)
        triangle = pd.DataFrame(index=DowJones.index)
        length = len(triangle.index)
        t = np.linspace(0, length, num=length)
        prices = minValue + maxValue / 2 * np.abs(signal.sawtooth(2 * np.pi * t / period))
        triangle['Open'] = prices
        triangle['High'] = prices
        triangle['Low'] = prices
        triangle['Close'] = prices
        triangle['Volume'] = 100000
        return triangle

class TradingEnv(gym.Env):
    def __init__(self, marketSymbol, startingDate, endingDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0, data_dir='./data/', features=['High', 'Low', 'Open', 'Close']):
        self.features = features
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)
        else:
            csvConverter = CSVHandler()
            csvName = "".join([data_dir, marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')
            if(exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            else:
                downloader1 = YahooFinance()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except Exception as e:
                    logging.error(f"Error in downloading data: {e}")
                if saving == True:
                    logging.info(f"Saving to CSV: {csvName}")
                    csvConverter.dataframeToCSV(csvName, self.data)
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.state = [self.data[feature][0:stateLength].tolist() for feature in self.features] + [[0]]
        self.reward = 0.
        self.done = 0
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1
        if startingPoint:
            self.setStartingPoint(startingPoint)

    def reset(self):
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.
        self.state = [self.data[feature][0:self.stateLength].tolist() for feature in self.features] + [[0]]
        self.reward = 0.
        self.done = 0
        self.t = self.stateLength
        self.numberOfShares = 0
        return self.state

    def computeLowerBound(self, cash, numberOfShares, price):
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def step(self, action):
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False
        if(action == 1):
            self.data['Position'][t] = 1
            if(self.data['Position'][t - 1] == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
        elif(action == 0):
            self.data['Position'][t] = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        self.t = self.t + 1

        self.state = [self.data[feature].iloc[self.t - self.stateLength : self.t].tolist() for feature in self.features] + [[self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1

        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        otherState = [self.data[feature][self.t - self.stateLength : self.t].tolist() for feature in self.features] + [[otherPosition]]
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}
        return self.state, self.reward, self.done, self.info

    def render(self, savePlots=False):
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        if savePlots:
            plt.savefig(''.join(['images/', str(self.marketSymbol), '_Rendering', '.png']))

    def setStartingPoint(self, startingPoint):
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))
        self.state = [self.data[feature][self.t - self.stateLength : self.t].tolist() for feature in self.features] + [[self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1

class TradingSimulator:
    def displayTestbench(self, startingDate, endingDate, features=['High', 'Low', 'Open', 'Close']):
        for _, stock in indices.items():
            env = TradingEnv(stock, startingDate, endingDate, 0, features=features)
            env.render()
        for _, stock in companies.items():
            env = TradingEnv(stock, startingDate, endingDate, 0, features=features)
            env.render()

    def analyseTimeSeries(self, stockName, startingDate, endingDate, splitingDate, features=['High', 'Low', 'Open', 'Close']):
        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]
        else:
            logging.info("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                logging.info("".join(['- ', stock]))
            for stock in indices:
                logging.info("".join(['- ', stock]))
            for stock in companies:
                logging.info("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")

        logging.info("\n\n\nAnalysis of the TRAINING phase time series")
        logging.info("------------------------------------------\n")
        trainingEnv = TradingEnv(stock, startingDate, splitingDate, 0, features=features)
        timeSeries = trainingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        logging.info("\n\n\nAnalysis of the TESTING phase time series")
        logging.info("------------------------------------------\n")
        testingEnv = TradingEnv(stock, splitingDate, endingDate, 0, features=features)
        timeSeries = testingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        logging.info("\n\n\nAnalysis of the entire time series (both training and testing phases)")
        logging.info("---------------------------------------------------------------------\n")
        tradingEnv = TradingEnv(stock, startingDate, endingDate, 0, features=features)
        timeSeries = tradingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

    def plotEntireTrading(self, trainingEnv, testingEnv, splitingDate, savePlots=False):
        ratio = trainingEnv.data['Money'][-1]/testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_')
        ax1.plot(data.loc[data['Action'] == 1.0].index,
                 data['Close'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(data.loc[data['Action'] == -1.0].index,
                 data['Close'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_')
        ax2.plot(data.loc[data['Action'] == 1.0].index,
                 data['Money'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(data.loc[data['Action'] == -1.0].index,
                 data['Money'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        ax1.axvline(pd.to_datetime(splitingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.to_datetime(splitingDate), color='black', linewidth=2.0)
        ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        if savePlots:
            plt.savefig(''.join(['images/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png']))

    def simulateNewStrategy(self, strategyName, stockName,
                            startingDate, endingDate, splitingDate,
                            observationSpace, actionSpace,
                            money, stateLength, transactionCosts,
                            bounds, step, numberOfEpisodes,
                            verbose=True, plotTraining=True, rendering=True, showPerformance=True,
                            saveStrategy=False,
                            savePlots=False,
                            data_dir='./data/',
                            strategies_dir='./models/',
                            features=['High', 'Low', 'Open', 'Close']):
        if(strategyName in models):
            strategy = models[strategyName]
            trainingParameters = [bounds, step]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            trainingParameters = [numberOfEpisodes]
            ai = True
        else:
            logging.info("The strategy specified is not valid, only the following models are supported:")
            for strategy in models:
                logging.info("".join(['- ', strategy]))
            for strategy in strategiesAI:
                logging.info("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]
        else:
            logging.info("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                logging.info("".join(['- ', stock]))
            for stock in indices:
                logging.info("".join(['- ', stock]))
            for stock in companies:
                logging.info("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")

        trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts, data_dir=data_dir, features=features)
        if ai:
            tradingStrategy = TDQN(observationSpace, actionSpace)
        else:
            if strategyName == 'Buy and Hold':
                tradingStrategy = BuyAndHold()
            elif strategyName == 'Sell and Hold':
                tradingStrategy = SellAndHold()
            elif strategyName == 'Trend Following Moving Averages':
                tradingStrategy = MovingAveragesTF()
            elif strategyName == 'Mean Reversion Moving Averages' :
                tradingStrategy = MovingAveragesMR()
            else:
                raise SystemError(strategyName)

        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,
                                               verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance,
                                               features=features)
        testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts, features=features)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance, features=features)
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv, splitingDate,savePlots=savePlots)
        if(saveStrategy):
            fileName = "".join([strategies_dir, strategy, "_", stock, "_", startingDate, "_", splitingDate])
            if ai:
                tradingStrategy.saveModel(fileName)
            else:
                fileHandler = open(fileName, 'wb')
                pickle.dump(tradingStrategy, fileHandler)
        return tradingStrategy, trainingEnv, testingEnv

    def simulateExistingStrategy(self, strategyName, stockName,
                                 startingDate, endingDate, splitingDate,
                                 observationSpace, actionSpace,
                                 money, stateLength, transactionCosts,
                                 rendering=True, showPerformance=True, strategiesDir='./models/', data_dir='./data/', features=['High', 'Low', 'Open', 'Close']):
        if(strategyName in models):
            strategy = models[strategyName]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            ai = True
        else:
            logging.info("The strategy specified is not valid, only the following models are supported:")
            for strategy in models:
                logging.info("".join(['- ', strategy]))
            for strategy in strategiesAI:
                logging.info("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]
        else:
            logging.info("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                logging.info("".join(['- ', stock]))
            for stock in indices:
                logging.info("".join(['- ', stock]))
            for stock in companies:
                logging.info("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")

        fileName = "".join([strategiesDir, strategy, "_", stock, "_", startingDate, "_", splitingDate])
        exists = os.path.isfile(fileName)
        if exists:
            if ai:
                tradingStrategy = TDQN(observationSpace, actionSpace)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb')
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")

        trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts, data_dir=data_dir, features=features)
        testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts, data_dir=data_dir, features=features)
        trainingEnv = tradingStrategy.testing(trainingEnv, trainingEnv, rendering=rendering, showPerformance=showPerformance)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance)
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv, splitingDate)
        return tradingStrategy, trainingEnv, testingEnv


class ReplayMemory:
    def __init__(self, capacity=capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
    def sample(self, batchSize):
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        return state, action, reward, nextState, done
    def __len__(self):
        return len(self.memory)
    def reset(self):
        self.memory = deque(maxlen=capacity)

class DQN(nn.Module):
    def __init__(self, numberOfInputs, numberOfOutputs, numberOfNeurons=numberOfNeurons, dropout=dropout):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(numberOfInputs, numberOfNeurons)
        self.fc2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc3 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc4 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.fc5 = nn.Linear(numberOfNeurons, numberOfOutputs)
        self.bn1 = nn.BatchNorm1d(numberOfNeurons)
        self.bn2 = nn.BatchNorm1d(numberOfNeurons)
        self.bn3 = nn.BatchNorm1d(numberOfNeurons)
        self.bn4 = nn.BatchNorm1d(numberOfNeurons)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
    def forward(self, input):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(input))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        output = self.fc5(x)
        return output

class TDQN:
    def __init__(self, observationSpace, actionSpace, numberOfNeurons=numberOfNeurons, dropout=dropout,
                 gamma=gamma, learningRate=learningRate, targetNetworkUpdate=targetNetworkUpdate,
                 epsilonStart=epsilonStart, epsilonEnd=epsilonEnd, epsilonDecay=epsilonDecay,
                 capacity=capacity, batchSize=batchSize):
        random.seed(0)
        self.device = torch.device('cuda:'+str(GPUNumber) if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.learningRate = learningRate
        self.targetNetworkUpdate = targetNetworkUpdate
        self.capacity = capacity
        self.batchSize = batchSize
        self.replayMemory = ReplayMemory(capacity)
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        self.policyNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork = DQN(observationSpace, actionSpace, numberOfNeurons, dropout).to(self.device)
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
        self.policyNetwork.eval()
        self.targetNetwork.eval()
        self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
        self.epsilonValue = lambda iteration: epsilonEnd + (epsilonStart - epsilonEnd) * math.exp(-1 * iteration / epsilonDecay)
        self.iterations = 0
        self.writer = SummaryWriter('runs/' + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S"))
    def getNormalizationCoefficients(self, tradingEnv):
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()
        coefficients = []
        margin = 1
        returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns)*margin)
        coefficients.append(coeffs)
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice)*margin)
        coefficients.append(coeffs)
        coeffs = (0, 1)
        coefficients.append(coeffs)
        coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
        coefficients.append(coeffs)
        return coefficients
    def processState(self, state, coefficients):
        closePrices = [state[0][i] for i in range(len(state[0]))]
        lowPrices = [state[1][i] for i in range(len(state[1]))]
        highPrices = [state[2][i] for i in range(len(state[2]))]
        volumes = [state[3][i] for i in range(len(state[3]))]

        returns = [(closePrices[i]-closePrices[i-1])/closePrices[i-1] for i in range(1, len(closePrices))]
        if coefficients[0][0] != coefficients[0][1]:
            state[0] = [((x - coefficients[0][0])/(coefficients[0][1] - coefficients[0][0])) for x in returns]
        else:
            state[0] = [0 for x in returns]
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(1, len(lowPrices))]
        if coefficients[1][0] != coefficients[1][1]:
            state[1] = [((x - coefficients[1][0])/(coefficients[1][1] - coefficients[1][0])) for x in deltaPrice]
        else:
            state[1] = [0 for x in deltaPrice]
        closePricePosition = []
        for i in range(1, len(closePrices)):
            deltaPrice = abs(highPrices[i]-lowPrices[i])
            if deltaPrice != 0:
                item = abs(closePrices[i]-lowPrices[i])/deltaPrice
            else:
                item = 0.5
            closePricePosition.append(item)
        if coefficients[2][0] != coefficients[2][1]:
            state[2] = [((x - coefficients[2][0])/(coefficients[2][1] - coefficients[2][0])) for x in closePricePosition]
        else:
            state[2] = [0.5 for x in closePricePosition]
        volumes = [volumes[i] for i in range(1, len(volumes))]
        if coefficients[3][0] != coefficients[3][1]:
            state[3] = [((x - coefficients[3][0])/(coefficients[3][1] - coefficients[3][0])) for x in volumes]
        else:
            state[3] = [0 for x in volumes]

        # The last feature is always the position.
        # By default they have 4 features and position:
        # [CLOSE, LOW, HIGH, VOLUME], xxx, [POSITION]
        # Where xxx are additional features we use.
        if len(state)-1 > 4:
            for idx in range(4, len(state) - 1):
                current_state = [state[idx][i] for i in range(len(state[idx]))]
                current_state = [current_state[i] for i in range(1, len(current_state))]
                state[idx] = [x for x in current_state]

        state = [item for sublist in state for item in sublist]

        return state
    def processReward(self, reward):
        return np.clip(reward, -rewardClipping, rewardClipping)
    def updateTargetNetwork(self):
        if(self.iterations % targetNetworkUpdate == 0):
            self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
    def chooseAction(self, state):
        with torch.no_grad():
            tensorState = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            QValues = self.policyNetwork(tensorState).squeeze(0)
            Q, action = QValues.max(0)
            action = action.item()
            Q = Q.item()
            QValues = QValues.cpu().numpy()
            return action, Q, QValues
    def chooseActionEpsilonGreedy(self, state, previousAction):
        if(random.random() > self.epsilonValue(self.iterations)):
            if(random.random() > alpha):
                action, Q, QValues = self.chooseAction(state)
            else:
                action = previousAction
                Q = 0
                QValues = [0, 0]
        else:
            action = random.randrange(self.actionSpace)
            Q = 0
            QValues = [0, 0]
        self.iterations += 1
        return action, Q, QValues
    def learning(self, batchSize=batchSize):
        if (len(self.replayMemory) >= batchSize):
            self.policyNetwork.train()
            state, action, reward, nextState, done = self.replayMemory.sample(batchSize)

            state = torch.tensor(state, dtype=torch.float, device=self.device)
            action = torch.tensor(action, dtype=torch.long, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            nextState = torch.tensor(nextState, dtype=torch.float, device=self.device)
            done = torch.tensor(done, dtype=torch.float, device=self.device)
            currentQValues = self.policyNetwork(state).gather(1, action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                nextActions = torch.max(self.policyNetwork(nextState), 1)[1]
                nextQValues = self.targetNetwork(nextState).gather(1, nextActions.unsqueeze(1)).squeeze(1)
                expectedQValues = reward + gamma * nextQValues * (1 - done)
            loss = F.smooth_l1_loss(currentQValues, expectedQValues)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), gradientClipping)
            self.optimizer.step()
            self.updateTargetNetwork()
            self.policyNetwork.eval()
    def training(self, trainingEnv, trainingParameters=[],
                 verbose=False, rendering=False, savePlots=False, plotTraining=False, showPerformance=False, features=['Open', 'High', 'Low', 'Close']):
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        if plotTraining:
            performanceTrain = []
            score = np.zeros((len(trainingEnvList), trainingParameters[0]))
            marketSymbol = trainingEnv.marketSymbol
            startingDate = trainingEnv.endingDate
            endingDate = '2020-1-1'
            money = trainingEnv.data['Money'][0]
            stateLength = trainingEnv.stateLength
            transactionCosts = trainingEnv.transactionCosts
            testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts, features=features)
            performanceTest = []
        try:
            if verbose:
                logging.info("Training progression (hardware selected => " + str(self.device) + "):")
            for episode in tqdm(range(trainingParameters[0]), disable=not(verbose)):
                for i in range(len(trainingEnvList)):
                    coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                    trainingEnvList[i].reset()
                    startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                    trainingEnvList[i].setStartingPoint(startingPoint)
                    state = self.processState(trainingEnvList[i].state, coefficients)
                    previousAction = 0
                    done = 0
                    stepsCounter = 0
                    if plotTraining:
                        totalReward = 0
                    while done == 0:
                        action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                        nextState, reward, done, info = trainingEnvList[i].step(action)
                        reward = self.processReward(reward)
                        nextState = self.processState(nextState, coefficients)
                        self.replayMemory.push(state, action, reward, nextState, done)
                        otherAction = int(not bool(action))
                        otherReward = self.processReward(info['Reward'])
                        otherNextState = self.processState(info['State'], coefficients)
                        otherDone = info['Done']
                        self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                        stepsCounter += 1
                        if stepsCounter == learningUpdatePeriod:
                            self.learning()
                            stepsCounter = 0
                        state = nextState
                        previousAction = action
                        if plotTraining:
                            totalReward += reward
                    if plotTraining:
                        score[i][episode] = totalReward
                if plotTraining:
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
        except KeyboardInterrupt:
            logging.info()
            logging.info("WARNING: Training prematurely interrupted...")
            logging.info()
            self.policyNetwork.eval()
        trainingEnv = self.testing(trainingEnv, trainingEnv)
        if rendering:
            trainingEnv.render()
        if plotTraining:
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot(performanceTrain)
            ax.plot(performanceTest)
            ax.legend(["Training", "Testing"])
            if savePlots:
                plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingPerformance', '.png']))
            for i in range(len(trainingEnvList)):
                self.plotTraining(score[i][:episode], marketSymbol)
        if showPerformance:
            analyser = PerformanceEstimator(trainingEnv.data)
            analyser.displayPerformance('TDQN')
        self.writer.close()
        return trainingEnv
    def testing(self, trainingEnv, testingEnv, rendering=False, savePlots=False, showPerformance=False, features=['Open', 'High', 'Low', 'Close']):
        dataAugmentation = DataAugmentation()
        testingEnvSmoothed = dataAugmentation.lowPassFilter(testingEnv, filterOrder)
        trainingEnv = dataAugmentation.lowPassFilter(trainingEnv, filterOrder)
        coefficients = self.getNormalizationCoefficients(trainingEnv)
        state = self.processState(testingEnvSmoothed.reset(), coefficients)
        testingEnv.reset()
        QValues0 = []
        QValues1 = []
        done = 0
        while done == 0:
            action, _, QValues = self.chooseAction(state)
            nextState, _, done, _ = testingEnvSmoothed.step(action)
            testingEnv.step(action)
            state = self.processState(nextState, coefficients)
            QValues0.append(QValues[0])
            QValues1.append(QValues[1])
        if rendering:
            testingEnv.render()
            self.plotQValues(QValues0, QValues1, testingEnv.marketSymbol)
        if showPerformance:
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance('TDQN Testing')
        return testingEnv
    def plotTraining(self, score, marketSymbol, savePlots=False):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
        ax1.plot(score)
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), 'TrainingResults', '.png']))
    def plotQValues(self, QValues0, QValues1, marketSymbol, savePlots=False):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Q values', xlabel='Time')
        ax1.plot(QValues0)
        ax1.plot(QValues1)
        ax1.legend(['Short', 'Long'])
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_QValues', '.png']))
    def plotExpectedPerformance(self, trainingEnv, trainingParameters=[], iterations=10, savePlots = False, features=['Open', 'High', 'Low', 'Close']):
        dataAugmentation = DataAugmentation()
        trainingEnvList = dataAugmentation.generate(trainingEnv)
        initialWeights =  copy.deepcopy(self.policyNetwork.state_dict())
        performanceTrain = np.zeros((trainingParameters[0], iterations))
        performanceTest = np.zeros((trainingParameters[0], iterations))
        marketSymbol = trainingEnv.marketSymbol
        startingDate = trainingEnv.endingDate
        endingDate = '2020-1-1'
        money = trainingEnv.data['Money'][0]
        stateLength = trainingEnv.stateLength
        transactionCosts = trainingEnv.transactionCosts
        testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts, features=features)
        logging.info("Hardware selected for training: " + str(self.device))
        try:
            for iteration in range(iterations):
                logging.info(''.join(["Expected performance evaluation progression: ", str(iteration+1), "/", str(iterations)]))
                for episode in tqdm(range(trainingParameters[0])):
                    for i in range(len(trainingEnvList)):
                        coefficients = self.getNormalizationCoefficients(trainingEnvList[i])
                        trainingEnvList[i].reset()
                        startingPoint = random.randrange(len(trainingEnvList[i].data.index))
                        trainingEnvList[i].setStartingPoint(startingPoint)
                        state = self.processState(trainingEnvList[i].state, coefficients)
                        previousAction = 0
                        done = 0
                        stepsCounter = 0
                        while done == 0:
                            action, _, _ = self.chooseActionEpsilonGreedy(state, previousAction)
                            nextState, reward, done, info = trainingEnvList[i].step(action)
                            reward = self.processReward(reward)
                            nextState = self.processState(nextState, coefficients)
                            self.replayMemory.push(state, action, reward, nextState, done)
                            otherAction = int(not bool(action))
                            otherReward = self.processReward(info['Reward'])
                            otherDone = info['Done']
                            otherNextState = self.processState(info['State'], coefficients)
                            self.replayMemory.push(state, otherAction, otherReward, otherNextState, otherDone)
                            stepsCounter += 1
                            if stepsCounter == learningUpdatePeriod:
                                self.learning()
                                stepsCounter = 0
                            state = nextState
                            previousAction = action
                    trainingEnv = self.testing(trainingEnv, trainingEnv)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performanceTrain[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performanceTrain[episode][iteration], episode)
                    testingEnv = self.testing(trainingEnv, testingEnv)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performanceTest[episode][iteration] = analyser.computeSharpeRatio()
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performanceTest[episode][iteration], episode)
                if iteration < (iterations-1):
                    trainingEnv.reset()
                    testingEnv.reset()
                    self.policyNetwork.load_state_dict(initialWeights)
                    self.targetNetwork.load_state_dict(initialWeights)
                    self.optimizer = optim.Adam(self.policyNetwork.parameters(), lr=learningRate, weight_decay=L2Factor)
                    self.replayMemory.reset()
                    self.iterations = 0
                    stepsCounter = 0
            iteration += 1
        except KeyboardInterrupt:
            logging.info()
            logging.info("WARNING: Expected performance evaluation prematurely interrupted...")
            logging.info()
            self.policyNetwork.eval()
        expectedPerformanceTrain = []
        expectedPerformanceTest = []
        stdPerformanceTrain = []
        stdPerformanceTest = []
        for episode in range(trainingParameters[0]):
            expectedPerformanceTrain.append(np.mean(performanceTrain[episode][:iteration]))
            expectedPerformanceTest.append(np.mean(performanceTest[episode][:iteration]))
            stdPerformanceTrain.append(np.std(performanceTrain[episode][:iteration]))
            stdPerformanceTest.append(np.std(performanceTest[episode][:iteration]))
        expectedPerformanceTrain = np.array(expectedPerformanceTrain)
        expectedPerformanceTest = np.array(expectedPerformanceTest)
        stdPerformanceTrain = np.array(stdPerformanceTrain)
        stdPerformanceTest = np.array(stdPerformanceTest)
        for i in range(iteration):
            fig = plt.figure()
            ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
            ax.plot([performanceTrain[e][i] for e in range(trainingParameters[0])])
            ax.plot([performanceTest[e][i] for e in range(trainingParameters[0])])
            ax.legend(["Training", "Testing"])
            if savePlots:
                plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingPerformance', str(i+1), '.png']))
        fig = plt.figure()
        ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
        ax.plot(expectedPerformanceTrain)
        ax.plot(expectedPerformanceTest)
        ax.fill_between(range(len(expectedPerformanceTrain)), expectedPerformanceTrain-stdPerformanceTrain, expectedPerformanceTrain+stdPerformanceTrain, alpha=0.25)
        ax.fill_between(range(len(expectedPerformanceTest)), expectedPerformanceTest-stdPerformanceTest, expectedPerformanceTest+stdPerformanceTest, alpha=0.25)
        ax.legend(["Training", "Testing"])
        if savePlots:
            plt.savefig(''.join(['images/', str(marketSymbol), '_TrainingTestingExpectedPerformance', '.png']))
        self.writer.close()
        return trainingEnv
    def saveModel(self, fileName):
        torch.save(self.policyNetwork.state_dict(), fileName)
    def loadModel(self, fileName):
        self.policyNetwork.load_state_dict(torch.load(fileName, map_location=self.device))
        self.targetNetwork.load_state_dict(self.policyNetwork.state_dict())
    def plotEpsilonAnnealing(self, savePlots=False):
        plt.figure()
        plt.plot([self.epsilonValue(i) for i in range(10*epsilonDecay)])
        plt.title("Plot Title")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon value")
        if savePlots:
            plt.savefig(''.join(['images/', 'EpsilonAnnealing', '.png']))

class PerformanceEstimator:
    def __init__(self, tradingData):
        self.data = tradingData
    def computePnL(self):
        self.PnL = self.data["Money"][-1] - self.data["Money"][0]
        return self.PnL
    def computeAnnualizedReturn(self):
        cumulativeReturn = self.data['Returns'].cumsum()
        cumulativeReturn = cumulativeReturn[-1]
        # Set time component to midnight with min.
        start = pd.to_datetime(self.data.index[0])
        end = pd.to_datetime(self.data.index[-1])
        timeElapsed = end - start
        timeElapsed = timeElapsed.days
        if(cumulativeReturn > -1):
            self.annualizedReturn = 100 * (((1 + cumulativeReturn) ** (365/timeElapsed)) - 1)
        else:
            self.annualizedReturn = -100
        return self.annualizedReturn
    def computeAnnualizedVolatility(self):
        self.annualizedVolatily = 100 * np.sqrt(252) * self.data['Returns'].std()
        return self.annualizedVolatily
    def computeSharpeRatio(self, riskFreeRate=0):
        expectedReturn = self.data['Returns'].mean()
        volatility = self.data['Returns'].std()
        if expectedReturn != 0 and volatility != 0:
            self.sharpeRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sharpeRatio = 0
        return self.sharpeRatio
    def computeSortinoRatio(self, riskFreeRate=0):
        expectedReturn = np.mean(self.data['Returns'])
        negativeReturns = [returns for returns in self.data['Returns'] if returns < 0]
        volatility = np.std(negativeReturns)
        if expectedReturn != 0 and volatility != 0:
            self.sortinoRatio = np.sqrt(252) * (expectedReturn - riskFreeRate)/volatility
        else:
            self.sortinoRatio = 0
        return self.sortinoRatio
    def computeMaxDrawdown(self, plotting=False, savePlots=False):
        capital = self.data['Money'].values
        through = np.argmax(np.maximum.accumulate(capital) - capital)
        if through != 0:
            peak = np.argmax(capital[:through])
            self.maxDD = 100 * (capital[peak] - capital[through])/capital[peak]
            self.maxDDD = through - peak
        else:
            self.maxDD = 0
            self.maxDDD = 0
            return self.maxDD, self.maxDDD
        if plotting:
            plt.figure(figsize=(10, 4))
            plt.plot(self.data['Money'], lw=2, color='Blue')
            plt.title("Plot Title")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.plot([self.data.iloc[[peak]].index, self.data.iloc[[through]].index],
                     [capital[peak], capital[through]], 'o', color='Red', markersize=5)
            plt.xlabel('Time')
            plt.ylabel('Price')
            if savePlots:
                plt.savefig(''.join(['images/', 'MaximumDrawDown', '.png']))
        return self.maxDD, self.maxDDD
    def computeProfitability(self):
        good = 0
        bad = 0
        profit = 0
        loss = 0
        index = next((i for i in range(len(self.data.index)) if self.data['Action'][i] != 0), None)
        if index == None:
            self.profitability = 0
            self.averageProfitLossRatio = 0
            return self.profitability, self.averageProfitLossRatio
        money = self.data['Money'][index]
        for i in range(index+1, len(self.data.index)):
            if(self.data['Action'][i] != 0):
                delta = self.data['Money'][i] - money
                money = self.data['Money'][i]
                if(delta >= 0):
                    good += 1
                    profit += delta
                else:
                    bad += 1
                    loss -= delta
        delta = self.data['Money'][-1] - money
        if(delta >= 0):
            good += 1
            profit += delta
        else:
            bad += 1
            loss -= delta
        self.profitability = 100 * good/(good + bad)
        if(good != 0):
            profit /= good
        if(bad != 0):
            loss /= bad
        if(loss != 0):
            self.averageProfitLossRatio = profit/loss
        else:
            self.averageProfitLossRatio = float('Inf')
        return self.profitability, self.averageProfitLossRatio
    def computeSkewness(self):
        self.skewness = self.data["Returns"].skew()
        return self.skewness
    def computePerformance(self):
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()
        self.performanceTable = [["Profit & Loss (P&L)", "{0:.0f}".format(self.PnL)],
                                 ["Annualized Return", "{0:.2f}".format(self.annualizedReturn) + '%'],
                                 ["Annualized Volatility", "{0:.2f}".format(self.annualizedVolatily) + '%'],
                                 ["Sharpe Ratio", "{0:.3f}".format(self.sharpeRatio)],
                                 ["Sortino Ratio", "{0:.3f}".format(self.sortinoRatio)],
                                 ["Maximum Drawdown", "{0:.2f}".format(self.maxDD) + '%'],
                                 ["Maximum Drawdown Duration", "{0:.0f}".format(self.maxDDD) + ' days'],
                                 ["Profitability", "{0:.2f}".format(self.profitability) + '%'],
                                 ["Ratio Average Profit/Loss", "{0:.3f}".format(self.averageProfitLossRatio)],
                                 ["Skewness", "{0:.3f}".format(self.skewness)]]
        return self.performanceTable
    def getComputedPerformance(self):
        self.computePnL()
        self.computeAnnualizedReturn()
        self.computeAnnualizedVolatility()
        self.computeProfitability()
        self.computeSharpeRatio()
        self.computeSortinoRatio()
        self.computeMaxDrawdown()
        self.computeSkewness()

        data = {
            'Metric': ["PnL",
                    "Annualized Return",
                    "Annualized Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Max Drawdown",
                    "Max Drawdown Duration (days)",
                    "Profitability",
                    "Avg Profit/Loss Ratio",
                    "Skewness"],
            'Value': [self.PnL,
                    self.annualizedReturn / 100,  # if expressed in %
                    self.annualizedVolatily / 100,  # if expressed in %
                    self.sharpeRatio,
                    self.sortinoRatio,
                    self.maxDD / 100,  # if expressed in %
                    self.maxDDD,
                    self.profitability / 100,  # if expressed in %
                    self.averageProfitLossRatio,
                    self.skewness]
        }

        return pd.DataFrame(data)

    def displayPerformance(self, name):
        self.computePerformance()
        headers = ["Performance Indicator", name]
        tabulation = tabulate(self.performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        logging.info('\n' + tabulation)