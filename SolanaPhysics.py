# SolanaPhysicsV26_Enhanced.py - Adaptive Harmonic Trading with Dynamic Fixes
# Enhanced version with all identified flaws corrected

import numpy as np
from pandas import DataFrame, merge
from datetime import datetime
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
import talib.abstract as ta

class SolanaPhysicsV26Enhanced(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # ROI - Gestaffelt für schnellere Gewinnmitnahmen
    minimal_roi = {
        "0": 0.08,    # 8% sofortiger Exit möglich
        "30": 0.05,   # Nach 30min: 5%
        "60": 0.03,   # Nach 1h: 3%
        "120": 0.015  # Nach 2h: 1.5%
    }
    
    stoploss = -0.08  # Basis-Stoploss
    use_custom_stoploss = True
    trailing_stop = False
    
    # Position Sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

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
        
        # === DYNAMISCHE PERCENTILE-BASIERTE FILTER ===
        # Flaw #1 Fix: Adaptive volatility and RSI thresholds
        dataframe['vol_percentile'] = dataframe['vol_ratio'].rolling(200).rank(pct=True)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_percentile'] = dataframe['rsi'].rolling(100).rank(pct=True)
        
        # === ZYKLUS-VALIDIERUNG ===
        # Flaw #2 Fix: Cycle strength validation
        dataframe['cycle_strength'] = ta.CORREL(
            dataframe['close'], 
            dataframe['sine'], 
            timeperiod=20  # Fixed window for correlation
        ).fillna(0)
        
        # === VOLUMEN-ANALYSEN ===
        # Flaw #3 Fix: Sustained volume instead of single-candle spikes
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['vol_ratio_raw'] = dataframe['volume'] / dataframe['vol_sma']
        
        # Sustained volume increase
        dataframe['vol_sustained'] = (
            (dataframe['vol_ratio_raw'] > 1.3) &
            (dataframe['vol_ratio_raw'].shift(1) > 1.2) &
            (dataframe['vol_ratio_raw'].shift(2) > 1.1)
        ).astype(int)
        
        # === MARKT-REGIME ERKENNUNG ===
        # Flaw #6 Fix: ADX-based market regime detection
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['regime'] = np.where(
            dataframe['adx'] > 25, 'trending',
            np.where(dataframe['adx'] < 20, 'ranging', 'transitional')
        )
        
        # === HÖHERER ZEITRAHMEN FILTER ===
        # Flaw #8 Fix: 1h timeframe confirmation
        htf_dataframe = self.dp.get_pair_dataframe(metadata['pair'], '1h')
        htf_dataframe['ema_50_1h'] = ta.EMA(htf_dataframe, timeperiod=50)
        htf_dataframe['rsi_1h'] = ta.RSI(htf_dataframe, timeperiod=14)
        
        # Merge HTF data
        dataframe = merge_informative_pair(
            dataframe, htf_dataframe, self.timeframe, '1h', ffill=True
        )
        
        # === TRENDFILTER (EMA-Abstand als "Potentielle Energie") ===
        # Flaw #4 Fix: ATR-normalized trend strength
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=50)
        
        # Trend-Stärke (ATR-normalisiert)
        dataframe['trend_strength_norm'] = (
            (dataframe['ema_fast'] - dataframe['ema_slow']) / 
            dataframe['atr']
        ).fillna(0)
        
        # === MOMENTUM (Beschleunigung) ===
        dataframe['rsi_ma'] = ta.SMA(dataframe['rsi'], timeperiod=5)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=3)
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Flaw #7 Fix: Multi-layered stoploss with volatility adjustment
        if current_profit < 0:
            multiplier = 2.5
        elif current_profit < 0.02:
            multiplier = 2.0
        else:
            multiplier = 1.2
            
        atr_stop = (last_candle['atr'] * multiplier) / current_rate
        
        # Volatility-adjusted maximum
        vol_adj_max = -max(0.08, last_candle['atr_percent'] / 100 * 2)
        
        # Pair-specific maximum (configurable)
        pair_max = self.config.get('pair_stoploss_max', {}).get(pair, -0.12)
        
        # Never exceed the most conservative limit
        return max(atr_stop, vol_adj_max, pair_max)

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                    current_rate: float, current_profit: float, **kwargs):
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
        trade_candles = int(trade_duration_min / 5)
        
        # Flaw #5 Fix: Tiered exit logic based on trade maturity
        if trade_duration_min < 30 and current_profit > 0.005:
            # Early exits: Small profit threshold
            if last_candle['sine'] < last_candle['leadsine'] - 0.15:
                return "early_wave_exit"
                
        elif 30 <= trade_duration_min < 120 and current_profit > 0.02:
            # Mature trades: Moderate profit threshold
            if last_candle['sine'] < last_candle['leadsine'] - 0.25:
                return "mature_wave_exit"
                
        elif trade_duration_min >= 120 and current_profit > 0.05:
            # Long-term holds: High profit threshold
            if last_candle['sine'] < last_candle['leadsine'] - 0.35:
                return "long_term_wave_exit"
        
        # === EXIT 1: Harmonischer Zyklus-Exit ===
        adaptive_exit_threshold = min(30, max(10, int(last_candle['dcperiod'] * 0.4)))
        
        if trade_candles > adaptive_exit_threshold:
            min_profit = max(0.008, last_candle['atr_percent'] / 100 * 0.5)
            
            if current_profit > min_profit:
                return "harmonic_cycle_mature"
        
        # === EXIT 2: Überhitzung (zu schnelle Bewegung) ===
        # Flaw #1 Fix: Use percentile-based threshold
        if (last_candle['vol_percentile'] > 0.95 and 
            last_candle['rsi_percentile'] > 0.95):
            if current_profit > 0.015:
                return "overheating_exit"
        
        # === EXIT 3: Trend-Break ===
        if (last_candle['close'] < last_candle['ema_trend'] and
            last_candle['trend_strength_norm'] < -0.5):
            
            if current_profit > 0:
                return "trend_break"
        
        return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # === HARMONISCHER LONG ENTRY ===
        # Flaw #1, #2, #3, #4, #6, #8 Fixes integrated
        
        # Base conditions
        conditions = [
            # 1. WELLENSIGNAL: Sine kreuzt Leadsine nach oben
            (dataframe['sine'] > dataframe['leadsine']) &
            (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
            
            # 2. WELLENSTÄRKE: Kreuzung nicht zu schwach
            (dataframe['sine'] - dataframe['leadsine'] > 0.05) &
            
            # 3. ZYKLUS-VALIDIERUNG: Starker Zyklus erkennbar
            (dataframe['cycle_strength'] > 0.3) &
            (dataframe['dcperiod'] > 12) &
            (dataframe['dcperiod'] < 40) &
            
            # 4. TRENDFILTER: Regime-basiert
            (
                # ENTWEDER: Trending Market
                ((dataframe['regime'] == 'trending') & 
                 (dataframe['trend_strength_norm'] > 0.3) &
                 (dataframe['close'] > dataframe['ema_trend'])) |
                
                # ODER: Ranging Market (stricter conditions)
                ((dataframe['regime'] == 'ranging') & 
                 (dataframe['vol_percentile'] < 0.5) &
                 (abs(dataframe['sine'] - dataframe['leadsine']) < 0.1) &
                 (dataframe['close'] > dataframe['ema_slow']))
            ) &
            
            # 5. VOLUMEN: Sustained volume
            (
                (dataframe['vol_sustained'] == 1) |
                (dataframe['vol_ratio_raw'] > 1.2)
            ) &
            
            # 6. HTF CONFIRMATION: Higher timeframe alignment
            (dataframe['close_1h'] > dataframe['ema_50_1h']) &
            (dataframe['rsi_1h'] > 40) &
            (dataframe['rsi_1h'] < 70) &
            
            # 7. MOMENTUM: RSI nicht überkauft
            (dataframe['rsi_percentile'] < 0.95) &
            (dataframe['rsi'] > 35) &
            (dataframe['rsi'] < 70) &
            
            # 8. VOLATILITÄTSFILTER: Adaptive Schwelle
            (dataframe['vol_percentile'] < 0.85)
        ]
        
        # Combine all conditions
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition & condition
            
        dataframe.loc[final_condition, 'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # Starkes physikalisches Exit-Signal
        dataframe.loc[
            (
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                
                # Kippung muss signifikant sein
                (dataframe['leadsine'] - dataframe['sine'] > 0.15) &
                
                # ODER: Trend dreht komplett
                (
                    (dataframe['leadsine'] > 0.4) |
                    ((dataframe['close'] < dataframe['ema_fast']) &
                     (dataframe['rsi'] < 45))
                )
            ),
            'exit_long'] = 1
        
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        
        # Flaw #10 Fix: Market hours awareness
        hour = current_time.hour
        day = current_time.weekday()
        
        # Skip low-liquidity periods (UTC)
        # Sunday 22:00 - Monday 06:00
        if day == 6 and hour >= 22:
            return False
        if day == 0 and hour < 6:
            return False
        
        # Friday 22:00 - Saturday 08:00
        if day == 4 and hour >= 22:
            return False
        if day == 5 and hour < 8:
            return False
        
        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        
        # Flaw #9 Fix: Volatility and cycle-adjusted position sizing
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Reduce stake for high volatility
        vol_factor = 1 / (1 + last_candle['vol_ratio_raw'])
        
        # Reduce stake for weak cycles
        cycle_factor = min(1, last_candle['cycle_strength'] * 2 + 0.5)
        
        # Reduce stake for ranging markets
        regime_factor = 1.0 if last_candle['regime'] == 'trending' else 0.7
        
        adjusted_stake = proposed_stake * vol_factor * cycle_factor * regime_factor
        
        return max(min_stake, min(adjusted_stake, max_stake))
