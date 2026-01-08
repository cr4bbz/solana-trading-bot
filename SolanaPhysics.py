# SolanaPhysics.py - Adaptive Version (ATR Stoploss & Cycle Filter)

import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysics(IStrategy):
        INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # 1. ROI: ZURÜCK ZUM GEWINNER
    # Wir waren zu gierig. 2% ist ein sicherer Hafen für Solana Scalps.
    minimal_roi = { 
        "0": 0.02,   # Ab 2% Gewinn wird verkauft. Punkt.
        "40": 0.01   # Wenn es länger dauert, nehmen wir auch 1%.
    }
    
    # 2. Stoploss (Der Not-Aus)
    stoploss = -0.10
    use_custom_stoploss = True  # Wir behalten die intelligente ATR-Berechnung!
    
    # 3. Trailing Stop (LOCKERER MACHEN)
    # Vorher: 0.005 (0.5%) -> Zu eng, fliegt sofort raus.
    # Neu: 0.01 (1.0%) -> Der Kurs darf 1% atmen, bevor wir Gewinne sichern.
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
        
        # (Optional: Volumen für spätere Analysen)
        dataframe['vol_mean'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rvol'] = dataframe['volume'] / dataframe['vol_mean']
        
        return dataframe

    # --- INTELLIGENTER STOPLOSS (Thermodynamik) ---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        # Wir holen uns die aktuelle Marktsituation
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Wir berechnen den Stop-Abstand basierend auf der aktuellen Hitze (ATR)
        # Formel: 2x die durchschnittliche Schwankungsbreite
        stop_distance = (last_candle['atr'] * 2) / current_rate
        
        # Sicherheits-Check:
        # Niemals enger als 1% (sonst werden wir sofort ausgestoppt)
        # Niemals weiter als 10% (Not-Aus)
        dynamic_stop = max(0.01, stop_distance)
        
        # Rückgabe muss negativ sein (z.B. -0.05)
        return max(-0.10, -dynamic_stop)

    # --- EINSTIEG (Kaufen) ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # A. KINEMATIK: Die perfekte Welle
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] < 0) &
                
                # B. KALIBRIERUNG: Nur handeln, wenn der Markt "sauber" schwingt
                # dcperiod < 10: Nur Rauschen (Noise) -> Nicht handeln
                # dcperiod > 40: Nur Trend (keine Welle) -> Nicht handeln
                (dataframe['dcperiod'] > 12) & 
                (dataframe['dcperiod'] < 40)
            ),
            'enter_long'] = 1
        return dataframe

    # --- AUSSTIEG (Verkaufen) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Verkauf am Wellenberg
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0)
            ),
            'exit_long'] = 1
        return dataframe
