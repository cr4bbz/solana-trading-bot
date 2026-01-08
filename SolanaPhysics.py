# SolanaPhysics.py - Fully Adaptive (Dynamic ROI + ATR Stop)

import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysics(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # 1. ROI (Nur noch Notfall-Fallschirm)
    minimal_roi = { 
        "0": 1.00, 
        "240": 0.01 # Erst nach 4 Stunden Not-Verkauf
    }
    
    # 2. STOPLOSS (Der einzig wahre Schutz)
    # Wir verlassen uns voll auf den ATR-Stoploss (unten im Code)
    stoploss = -0.10
    use_custom_stoploss = True
    
    # 3. TRAILING STOP -> AUSGESCHALTET!
    # Wir wollen nicht mehr bei kleinen Wacklern rausfliegen.
    trailing_stop = False
    
    # (Die Parameter hier sind jetzt egal, weil False, aber wir lassen sie stehen)
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    
    # --- INDIKATOREN ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Zeit (Frequenz)
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # Bewegung (Welle)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # Hitze (Volatilität)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    # --- DYNAMISCHER STOPLOSS (ATR) ---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 2x ATR Abstand
        stop_distance = (last_candle['atr'] * 2) / current_rate
        dynamic_stop = max(0.01, stop_distance)
        return max(-0.10, -dynamic_stop)

    # --- DYNAMISCHER EXIT (Die Innovation) ---
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Wir holen uns die aktuelle Zyklus-Länge des Marktes (in Kerzen)
        # z.B. 20 Kerzen
        current_cycle = last_candle['dcperiod']
        
        # Wie viele Kerzen sind wir schon im Trade?
        # (Aktuelle Zeit - Startzeit) / 5 Minuten
        trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_candles = trade_duration_min / 5
        
        # LOGIK:
        # Wenn wir länger als ein halber Zyklus drin sind...
        # ...sollten wir eigentlich am Gipfel sein.
        # Wenn wir dann im Plus sind (> 0.5%) -> RAUS!
        if (trade_candles > (current_cycle / 2)) and (current_profit > 0.005):
            return "dynamic_cycle_exit"
            
        return None

    # --- EINSTIEG ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] < 0) &
                (dataframe['dcperiod'] > 12) & 
                (dataframe['dcperiod'] < 40)
            ),
            'enter_long'] = 1
        return dataframe

    # --- AUSSTIEG (Signal) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
