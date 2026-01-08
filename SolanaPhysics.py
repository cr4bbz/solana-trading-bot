# SolanaPhysicsV25.py - Fully Adaptive (Harmonic Alignment)

import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysicsV25(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # 1. ROI (Notfall-Fallschirm weit oben)
    minimal_roi = { "0": 10.0 } # Wir wollen den Exit über die Physik regeln
    
    # 2. STOPLOSS (Der Fels - wird durch custom_stoploss dynamisch)
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    # --- INDIKATOREN ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Zeit (Frequenz/Wellenlänge)
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # Bewegung (Sinus-Phase)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # Hitze & Energie
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        
        return dataframe

    # --- DYNAMISCHER STOPLOSS (ATR & Profit-Kopplung) ---
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Die Bedingung: Je höher der Profit, desto enger der ATR-Schutz.
        # Im Minus: 3x ATR (Raum zum Atmen). Im Plus: 1.5x ATR (Sicherung).
        multiplier = 3.0 if current_profit < 0.01 else 1.5
        stop_distance = (last_candle['atr'] * multiplier) / current_rate
        
        return max(-0.10, -stop_distance)

    # --- DYNAMISCHER EXIT (Zyklus-Kopplung) ---
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Kopplung an die Wellenlänge (DCPeriod)
        trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_candles = trade_duration_min / 5
        
        # Wenn die "Halbwertszeit" der Welle überschritten ist...
        if (trade_candles > (last_candle['dcperiod'] / 2)):
            # ...und wir einen harmonischen Gewinn haben (relativ zur ATR)
            atr_profit_threshold = (last_candle['atr'] / current_rate)
            if current_profit > max(0.005, atr_profit_threshold):
                return "harmonic_cycle_exit"
            
        return None

    # --- HARMONISCHER EINSTIEG ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Hier bedingen sich die Werte gegenseitig:
        # 1. Das Signal: Sinus-Kreuzung (Kinematik)
        # 2. Die Energie: Erforderliches Volumen steigt mit der Reife (Leadsine)
        #    Frühe Welle (Leadsine klein) -> Wenig Volumen nötig.
        #    Späte Welle (Leadsine groß) -> Viel Volumen nötig.
        
        dataframe.loc[
            (
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                
                # DIE KOPPLUNG: Volumenbedarf = SMA * (1 + Leadsine)
                # Wenn Leadsine -0.5 ist -> SMA * 0.5 nötig.
                # Wenn Leadsine +0.5 ist -> SMA * 1.5 nötig.
                (dataframe['volume'] > (dataframe['vol_sma'] * (1.0 + dataframe['leadsine']))) &
                
                # Stabilitätsfilter: Welle muss klar erkennbar sein
                (dataframe['dcperiod'] > 10)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Der Exit-Trend ist nur noch das ultimative physikalische Stopp-Signal
        dataframe.loc[
            (
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0.5) # Nur wenn die Welle wirklich kippt
            ),
            'exit_long'] = 1
        return dataframe
