# SolanaPhysics.py - Relaxed Version (Only Sine Wave)

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class SolanaPhysics(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # ROI: Ziemlich konservativ, um Gewinne schnell zu sichern
    minimal_roi = { 
        "0": 0.02, 
        "20": 0.01 
    }
    
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Cluster 0: Kalibrierung
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # Cluster 1: Kinematik (Die Welle)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # Wir berechnen RVOL und ATR trotzdem für die Analyse
        dataframe['vol_mean'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rvol'] = dataframe['volume'] / dataframe['vol_mean']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # NUR NOCH DIE WELLE (Kinematik) - Der Scharfschütze
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] < 0)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Verkauf beim Kreuzen oben
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
