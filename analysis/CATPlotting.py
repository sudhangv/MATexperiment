import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#df = pd.read_csv('CATMeasurements/WavemeterLogs/153mA38.0C.csv', skiprows=2) 
df = pd.read_csv('CATMeasurements/WavemeterLogs/111.5mA39.8C.csv', skiprows=1) 

plt.gcf().set_dpi(200)
plt.plot(df['Lambda (THz)'], 'o')
plt.xticks(np.arange(0,1751, 175), labels=np.arange(0,10.1, 1))
plt.xlabel('Time (minutes)')
plt.ylabel('Frequnecy (Ghz)')
plt.grid()
    