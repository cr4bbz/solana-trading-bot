# SolanaPhysicsV26_Fixed.py - Fixed Adaptive Harmonic Trading
import numpy as np
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
import talib.abstract as ta
import logging

logger = logging.getLogger(__name__)

class SolanaPhysicsV26Fixed(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.08,
        "30": 0.05,
        "60": 0.03,
        "120": 0.015
    }
    
    stoploss = -0.08
    use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = True
    max_entry_position_adjustment = 0
    
    # Required for informative pairs
    informative_pairs = []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core harmonic indicators
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']
        
        # Volatility metrics
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Volume analysis
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['vol_ratio_raw'] = dataframe['volume'] / dataframe['vol_sma']
        
        # Sustained volume (Fix #3)
        dataframe['vol_sustained'] = (
            (dataframe['vol_ratio_raw'] > 1.3) &
            (dataframe['vol_ratio_raw'].shift(1) > 1.2) &
            (dataframe['vol_ratio_raw'].shift(2) > 1.1)
        ).astype(int)
        
        # Dynamic percentiles (Fix #1)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['vol_percentile'] = dataframe['vol_ratio_raw'].rolling(200, min_periods=20).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
        ).fillna(0.5)
        
        dataframe['rsi_percentile'] = dataframe['rsi'].rolling(100, min_periods=20).apply(
            lambda x: x.rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
        ).fillna(0.5)
        
        # Cycle validation (Fix #2)
        dataframe['cycle_strength'] = 0.0
        for i in range(len(dataframe)):
            if i >= 20:
                period = int(dataframe['dcperiod'].iloc[i])
                if period >= 10 and period <= 50:
                    corr = dataframe['close'].iloc[i-period:i].corr(
                        dataframe['sine'].iloc[i-period:i]
                    )
                    dataframe.loc[dataframe.index[i], 'cycle_strength'] = corr if not np.isnan(corr) else 0.0
        
        # Market regime (Fix #6)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['regime'] = np.where(
            dataframe['adx'] > 25, 'trending',
            np.where(dataframe['adx'] < 20, 'ranging', 'transitional')
        )
        
        # Trend indicators (Fix #4 - ATR normalized)
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=50)
        
        dataframe['trend_strength_norm'] = (
            (dataframe['ema_fast'] - dataframe['ema_slow']) / 
            dataframe['atr'].replace(0, 0.001)  # Avoid division by zero
        ).fillna(0)
        
        # Higher timeframe data (Fix #8)
        try:
            htf_dataframe = self.dp.get_pair_dataframe(metadata['pair'], '1h')
            htf_dataframe['ema_50_1h'] = ta.EMA(htf_dataframe, timeperiod=50)
            htf_dataframe['rsi_1h'] = ta.RSI(htf_dataframe, timeperiod=14)
            
            dataframe = merge_informative_pair(
                dataframe, htf_dataframe, self.timeframe, '1h', ffill=True
            )
        except Exception as e:
            logger.warning(f"Could not get 1h data for {metadata['pair']}: {e}")
            dataframe['ema_50_1h'] = dataframe['ema_trend']
            dataframe['rsi_1h'] = 50
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Basic entry conditions
        conditions = [
            # Wave signal
            (dataframe['sine'] > dataframe['leadsine']) &
            (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
            
            # Cycle validation (Fix #2)
            (dataframe['cycle_strength'] > 0.3) &
            (dataframe['dcperiod'] > 12) &
            (dataframe['dcperiod'] < 40) &
            
            # Trend filter
            (dataframe['close'] > dataframe['ema_trend']) &
            (dataframe['trend_strength_norm'] > -0.5) &
            
            # Volume confirmation (Fix #3)
            (
                (dataframe['vol_sustained'] == 1) |
                (dataframe['vol_ratio_raw'] > 1.2)
            ) &
            
            # Dynamic filters (Fix #1)
            (dataframe['vol_percentile'] < 0.85) &
            (dataframe['rsi_percentile'] < 0.95) &
            (dataframe['rsi'] > 35) &
            (dataframe['rsi'] < 70) &
            
            # Higher timeframe confirmation (Fix #8)
            (dataframe['close'] > dataframe['ema_50_1h']) &
            (dataframe['rsi_1h'] > 40) &
            (dataframe['rsi_1h'] < 70)
        ]
        
        dataframe.loc[conditions[0], 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] - dataframe['sine'] > 0.15)
            ),
            'exit_long'] = 1
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return -0.08
                
            last_candle = dataframe.iloc[-1]
            
            # ATR-based dynamic stop (Fix #7)
            if current_profit < 0:
                multiplier = 2.5
            elif current_profit < 0.02:
                multiplier = 2.0
            else:
                multiplier = 1.2
                
            atr_stop = (last_candle['atr'] * multiplier) / current_rate
            
            # Maximum limits
            vol_adj_max = -max(0.08, last_candle['atr_percent'] / 100 * 2)
            pair_max = self.config.get('pair_stoploss_max', {}).get(pair, -0.12)
            
            return max(atr_stop, vol_adj_max, pair_max)
            
        except Exception as e:
            logger.error(f"Error in custom_stoploss: {e}")
            return -0.08

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return None
                
            last_candle = dataframe.iloc[-1]
            trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # Tiered exits (Fix #5)
            if trade_duration_min < 30 and current_profit > 0.005:
                if last_candle['sine'] < last_candle['leadsine'] - 0.15:
                    return "early_wave_exit"
                    
            elif 30 <= trade_duration_min < 120 and current_profit > 0.02:
                if last_candle['sine'] < last_candle['leadsine'] - 0.25:
                    return "mature_wave_exit"
                    
            elif trade_duration_min >= 120 and current_profit > 0.05:
                if last_candle['sine'] < last_candle['leadsine'] - 0.35:
                    return "long_term_wave_exit"
            
            # Overheating exit (Fix #1)
            if last_candle['vol_percentile'] > 0.95 and last_candle['rsi_percentile'] > 0.95:
                if current_profit > 0.015:
                    return "overheating_exit"
            
            return None
            
        except Exception as e:
            logger.error(f"Error in custom_exit: {e}")
            return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: str,
                           side: str, **kwargs) -> bool:
        
        # Market hours filter (Fix #10)
        hour = current_time.hour
        day = current_time.weekday()
        
        # Skip low-liquidity periods
        if (day == 6 and hour >= 22) or (day == 0 and hour < 6):
            return False
        if (day == 4 and hour >= 22) or (day == 5 and hour < 8):
            return False
            
        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        
        try:
            # Position sizing (Fix #9)
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return proposed_stake
                
            last_candle = dataframe.iloc[-1]
            
            vol_factor = 1 / (1 + last_candle['vol_ratio_raw'])
            cycle_factor = min(1, last_candle['cycle_strength'] * 2 + 0.5)
            regime_factor = 1.0 if last_candle['regime'] == 'trending' else 0.7
            
            adjusted_stake = proposed_stake * vol_factor * cycle_factor * regime_factor
            
            return max(min_stake, min(adjusted_stake, max_stake))
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake
