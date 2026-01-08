# SolanaPhysicsV26.py - Optimized Harmonic Trading

import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysicsV26(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # ROI - Gestaffelt für schnellere Gewinnmitnahmen
    minimal_roi = {
        "0": 0.08,    # 8% sofortiger Exit möglich
        "30": 0.05,   # Nach 30min: 5%
        "60": 0.03,   # Nach 1h: 3%
        "120": 0.015  # Nach 2h: 1.5%
    }
    
    stoploss = -0.08  # Enger, aber mit dynamischer Anpassung
    use_custom_stoploss = True
    trailing_stop = False
    
    # Position Sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 0  # Erstmal kein DCA

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === PHYSIKALISCHE BASIS ===
        # Dominante Zykluslänge
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # Wellenbewegung (Sine/Cosine-Dekomposition)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # === ENERGIE & VOLATILITÄT ===
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Volatilitäts-Normalisierung (für adaptive Schwellen)
        dataframe['atr_ma'] = ta.SMA(dataframe['atr_percent'], timeperiod=50)
        dataframe['vol_ratio'] = dataframe['atr_percent'] / dataframe['atr_ma']
        
        # === VOLUMEN-ANALYSEN ===
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['vol_sma']
        
        # Relative Volume Strength
        dataframe['vol_surge'] = (
            (dataframe['volume'] > dataframe['vol_sma'] * 1.3) &
            (dataframe['volume'].shift(1) <= dataframe['vol_sma'].shift(1) * 1.3)
        ).astype(int)
        
        # === TRENDFILTER (EMA-Abstand als "Potentielle Energie") ===
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=50)
        
        # Trend-Stärke (normalisiert)
        dataframe['trend_strength'] = (
            (dataframe['ema_fast'] - dataframe['ema_slow']) / dataframe['close']
        ) * 100
        
        # === MOMENTUM (Beschleunigung) ===
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=5)
        
        # Momentum-Divergenz (Beschleunigungsänderung)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=3)
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # ATR-basierter dynamischer Stop
        # Im Minus: 2.5x ATR (mehr Raum)
        # Im Plus: 1.2x ATR (Trailing-ähnlich)
        if current_profit < 0:
            multiplier = 2.5
        elif current_profit < 0.02:
            multiplier = 2.0
        else:
            multiplier = 1.2
            
        atr_stop = (last_candle['atr'] * multiplier) / current_rate
        
        # Nie weiter als initialer Stop (-8%)
        return max(-0.08, -atr_stop)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_candles = int(trade_duration_min / 5)
        
        # === EXIT 1: Harmonischer Zyklus-Exit ===
        # Statt DCPeriod/2 nutzen wir eine adaptive Schwelle basierend auf Volatilität
        adaptive_exit_threshold = min(30, max(10, int(last_candle['dcperiod'] * 0.4)))
        
        if trade_candles > adaptive_exit_threshold:
            # Minimaler Profit basierend auf ATR
            min_profit = max(0.008, last_candle['atr_percent'] / 100 * 0.5)
            
            if current_profit > min_profit:
                return "harmonic_cycle_mature"
        
        # === EXIT 2: Wellenkippung (Physikalisches Signal) ===
        # Sine kippt unter Leadsine UND Momentum bricht ein
        if (last_candle['sine'] < last_candle['leadsine'] and 
            last_candle['sine'] - last_candle['leadsine'] < -0.15):
            
            if current_profit > 0.002:  # Mindestens breakeven
                return "wave_reversal"
        
        # === EXIT 3: Überhitzung (zu schnelle Bewegung) ===
        if last_candle['vol_ratio'] > 3.0 and last_candle['rsi'] > 78:
            if current_profit > 0.015:
                return "overheating_exit"
        
        # === EXIT 4: Trend-Break ===
        if (last_candle['close'] < last_candle['ema_trend'] and
            last_candle['trend_strength'] < -0.3):
            
            if current_profit > 0:
                return "trend_break"
        
        return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # === HARMONISCHER LONG ENTRY ===
        dataframe.loc[
            (
                # 1. WELLENSIGNAL: Sine kreuzt Leadsine nach oben
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                
                # 2. WELLENSTÄRKE: Kreuzung nicht zu schwach (vermeidet Whipsaws)
                (dataframe['sine'] - dataframe['leadsine'] > 0.05) &
                
                # 3. TRENDFILTER: Preis über mittelfristigem Trend
                (dataframe['close'] > dataframe['ema_trend']) &
                (dataframe['trend_strength'] > -0.2) &
                
                # 4. VOLUMEN: Adaptiv an Wellenlage
                # Statt (1 + leadsine) nutzen wir moderate Schwelle
                (
                    # ENTWEDER: Volumen-Surge (starker Impuls)
                    (dataframe['vol_surge'] == 1) |
                    # ODER: Erhöhtes Volumen (mindestens 1.2x)
                    (dataframe['vol_ratio'] > 1.2)
                ) &
                
                # 5. ZYKLUS-KLARHEIT: Dominanter Zyklus erkennbar
                (dataframe['dcperiod'] > 12) &
                (dataframe['dcperiod'] < 40) &
                
                # 6. MOMENTUM: RSI nicht überkauft
                (dataframe['rsi'] > 35) &
                (dataframe['rsi'] < 70) &
                
                # 7. VOLATILITÄTSFILTER: Nicht in extremer Volatilität
                (dataframe['vol_ratio'] < 2.5)
            ),
            'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Starkes physikalisches Exit-Signal
        # Nur bei klarer Wellenkippung
        dataframe.loc[
            (
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                
                # Kippung muss signifikant sein
                (dataframe['leadsine'] - dataframe['sine'] > 0.15) &
                
                # ODER: Trend dreht komplett
                (
                    (dataframe['leadsine'] > 0.4) |
                    (dataframe['close'] < dataframe['ema_fast']) &
                    (dataframe['rsi'] < 45)
                )
            ),
            'exit_long'] = 1
        
        return dataframe
