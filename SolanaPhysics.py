# SolanaPhysics.py - Adaptive Physics Strategy
# Architektur: 4 Cluster (Kinematik, Dynamik, Thermodynamik, Struktur)

import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SolanaPhysics(IStrategy):
    INTERFACE_VERSION = 3
    
    # 1. Grund-Konfiguration
    # ---------------------
    timeframe = '5m'  # Der Herzschlag (5 Minuten)
    
    # ROI: Wann nehmen wir Gewinn mit?
    # Dynamisch: Zuerst gierig (+4%), später bescheiden
    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "60": 0.01
    }
    
    # Stoploss: Der Not-Aus
    stoploss = -0.05  # Hartes Limit bei -5%
    
    # Trailing Stop: Gewinne absichern
    trailing_stop = True
    trailing_stop_positive = 0.005  # Ab 0.5% Gewinn aktiv
    trailing_stop_positive_offset = 0.015
    
    # 2. Indikatoren-Berechnung (Die Physik)
    # --------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # --- CLUSTER 0: KALIBRIERUNG (Das Gehirn) ---
        # Wir messen die Eigenfrequenz des Marktes
        dataframe['dcperiod'] = ta.HT_DCPERIOD(dataframe)
        
        # --- CLUSTER 1: KINEMATIK (Bewegung) ---
        # Hilbert Sine Wave: Findet Wendepunkte im Zyklus
        dataframe['sine'], dataframe['leadsine'] = ta.HT_SINE(dataframe)
        
        # --- CLUSTER 2: DYNAMIK (Masse) ---
        # RVOL (Relatives Volumen): Ist das Volumen höher als der Durchschnitt?
        # Wir nutzen ein 24-Perioden Fenster (ca. 2 Stunden) als Referenz
        dataframe['vol_mean'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rvol'] = dataframe['volume'] / dataframe['vol_mean']
        
        # --- CLUSTER 3: THERMODYNAMIK (Risiko) ---
        # ATR (Volatilität)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    # 3. Kauf-Logik (Entry)
    # ---------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # A. KINEMATIK: Perfektes Timing im Wellental
                (dataframe['sine'] > dataframe['leadsine']) &
                (dataframe['sine'].shift(1) <= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] < 0) & # Nur kaufen, wenn Welle unten ist
                
                # B. DYNAMIK: Masse muss da sein (Bestätigung)
                (dataframe['rvol'] > 1.2) &
                
                # C. STRUKTUR: Kein Kauf im freien Fall (Preis höher als vor 1h)
                (dataframe['close'] > dataframe['close'].shift(12))
            ),
            'enter_long'] = 1

        return dataframe

    # 4. Verkauf-Logik (Exit)
    # -----------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Verkauf am Wellenberg
                (dataframe['sine'] < dataframe['leadsine']) &
                (dataframe['sine'].shift(1) >= dataframe['leadsine'].shift(1)) &
                (dataframe['leadsine'] > 0) # Nur verkaufen, wenn Welle oben ist
            ),
            'exit_long'] = 1
            
        return dataframe
