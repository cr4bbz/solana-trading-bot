# SolanaPhysics.py - Adaptive Sniperedition
# Strategie: Hilbert Sine Wave + ATR Stoploss + Cycle Filter
# Ziel: Hohe Winrate durch konservative Exits

import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysics(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # 1. ROI: ZURÜCK ZUM GEWINNER (Konservativ)
    # Wir nehmen Gewinne schnell mit, bevor sie verschwinden.
    minimal_roi = { 
        "0": 0.02,   # 2% Gewinn? Sofort einsacken.
        "40": 0.01   # Dauert es länger als 40 Min? Dann reichen auch 1%.
    }
    
    # 2. STOPLOSS (Thermodynamik)
    # Hardcap bei -10%, aber der echte Stop wird unten per ATR berechnet.
    stoploss = -0.10
    use_custom_stoploss = True
    
    # 3. TRAILING STOP (Raum zum Atmen)
    # Wir lassen dem Kurs 1% Luft, bevor wir absichern.
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    
    # --- INDIKATOREN ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Cluster 0: Kalibrierung (Zyklusdauer messen)
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # Cluster 1: Kinematik (Die Welle)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # Cluster 3: Thermodynamik (Volatilität messen für Stoploss)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Hilfsdaten für Analyse (Volumen)
        dataframe['vol_mean'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rvol'] = dataframe['volume'] / dataframe['vol_mean']
        
        return dataframe

    # --- INTELLIGENTER STOPLOSS (ATR) ---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        # Daten holen
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Stop-Abstand: 2x ATR (Markt-Temperatur)
        stop_distance = (last_candle['atr'] * 2) / current_rate
        
        # Sicherheits-Regeln:
        # Nicht enger als 1% (sonst fliegt man durch Rauschen raus)
        dynamic_stop = max(0.01, stop_distance)
        
        # Rückgabe muss negativ sein
        return max(-0.10, -dynamic_stop)

    # --- EINSTIEG (Entry) ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # A. KINEMATIK: Welle kreuzt unten nach oben
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] < 0) &
                
                # B. FILTER: Nur in stabilen Zyklen handeln
                (dataframe['dcperiod'] > 12) & 
                (dataframe['dcperiod'] < 40)
            ),
            'enter_long'] = 1
        return dataframe

    # --- AUSSTIEG (Exit) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # A. KINEMATIK: Welle kreuzt oben nach unten
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
