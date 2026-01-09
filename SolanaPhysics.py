# SolanaPhysicsV39.py - THE PHYSICAL SINGULARITY
# Integration: Entropy, Hurst Exponent, Efficiency Ratio & Hilbert Transform

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

class SolanaPhysicsV39(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    
    # ROI & Stoploss: Diszipliniert und an V33 angelehnt
    minimal_roi = { "0": 0.10, "120": 0.03, "360": 0.015 }
    stoploss = -0.05 
    use_custom_stoploss = True
    trailing_stop = False 

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. KINEMATIK (Hilbert Transform)
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        hilbert = ta.HT_SINE(dataframe)
        dataframe['sine'] = hilbert['sine']
        dataframe['leadsine'] = hilbert['leadsine']

        # 2. FRAKTALE DYNAMIK (Hurst-Exponent Schätzung)
        # H > 0.5: Trend-persistent | H < 0.5: Rauschen/Mean-Reversion
        window = 20
        rolling_max = dataframe['high'].rolling(window).max()
        rolling_min = dataframe['low'].rolling(window).min()
        rolling_std = dataframe['close'].rolling(window).std()
        # Rescaled Range Schätzung (vereinfacht für Echtzeit)
        dataframe['hurst'] = np.log((rolling_max - rolling_min) / (rolling_std + 1e-9)) / np.log(window)

        # 3. INFORMATIONSTHEORIE (Shannon Entropie)
        # Misst die Unordnung im Preisrauschen
        def calculate_entropy(series, period=20):
            # Berechnet die Wahrscheinlichkeitsverteilung der Preisänderungen
            return series.rolling(period).apply(
                lambda x: -np.sum(np.histogram(x, bins=10, density=True)[0] * np.log(np.histogram(x, bins=10, density=True)[0] + 1e-9))
            )
        dataframe['entropy'] = calculate_entropy(dataframe['close'].pct_change())
        dataframe['entropy_sma'] = ta.SMA(dataframe['entropy'], timeperiod=10)

        # 4. WIRKUNGSGRAD (Efficiency Ratio nach Kaufman)
        # Verhältnis von Netto-Weg zu Brutto-Weg
        net_change = abs(dataframe['close'] - dataframe['close'].shift(window))
        sum_of_changes = abs(dataframe['close'] - dataframe['close'].shift(1)).rolling(window).sum()
        dataframe['efficiency'] = net_change / (sum_of_changes + 1e-9)

        # 5. BASIS TREND & VOLUMEN
        dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)
        dataframe['vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # Erst wenn wir physikalische Arbeit geleistet haben (>1.6% Profit), sichern wir ab.
        if current_profit > 0.016:
            return 0.002
        return 1.0

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            trade_duration_min = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # THERMODYNAMISCHER EXIT: Wenn die Entropie (Unordnung) wieder massiv steigt
            # oder die Sinus-Welle am Peak kippt.
            if (last_candle['sine'] < last_candle['leadsine']) and (current_profit > 0.008):
                return "physical_peak_exit"

            # Zeitliche Begrenzung zur Kapitalsicherung
            if trade_duration_min > 1200 and current_profit < 0.002:
                return "efficiency_timeout"
            
        return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # DIE PHYSIKALISCHE SINGULARITÄT:
        # Der Einstieg erfolgt nur, wenn Ordnung aus dem Chaos entsteht.
        dataframe.loc[
            (
                # A. ORDNUNG: Sinkende Entropie (Markt wird "klarer")
                (dataframe['entropy'] < dataframe['entropy_sma']) &
                
                # B. PERSISTENZ: Hurst > 0.5 (Der Trend will sich fortsetzen)
                (dataframe['hurst'] > 0.52) &
                
                # C. WIRKUNGSGRAD: Preisbewegung ist effizient (>30% Nutzarbeit)
                (dataframe['efficiency'] > 0.3) &
                
                # D. KINEMATIK: Sinus-Kreuzung im Aufwärtstrend
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                
                # E. ENERGIE: Volumenbestätigung
                (dataframe['volume'] > dataframe['vol_sma'] * 0.8)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[[], 'exit_long'] = 1
        return dataframe
