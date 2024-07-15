import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#신호를 0초에 보냈을 때, reinforced와 hole비교

def time_revision(time, ch2):
    no_signal = ch2[:100]
    max = np.max(no_signal)
    min = np.min(no_signal)

    if max >=0:
        max=round(max,3)
    else:
        max=-round(-max,3)

    if min>=0:
        min = round(min,3)
    else:
        min=-round(-min,3)

    c = 0.0005

    indices = np.where((ch2>max+c)|(ch2<min-c))
    index = indices[0][0]
    start_time = time[index]
    time_revision = [x-start_time for x in time]

    return time_revision


r = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석_240411_ver6\\초음파신호 ToF 분석\\raw_data\\ap\\r\\240402_tek0000ALL.csv"
h = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석_240411_ver6\\초음파신호 ToF 분석\\raw_data\\ap\\h\\240402_tek0100ALL.csv"


data_r = pd.read_csv(r, skiprows=20)
time_r = data_r['TIME']
ch1_r = data_r['CH1']
ch2_r = data_r['CH2']

time_revision_r = time_revision(time_r, ch2_r)

data_h = pd.read_csv(h, skiprows=20)
time_h = data_h['TIME']
ch1_h = data_h['CH1']
ch2_h = data_h['CH2']

time_revision_h = time_revision(time_h, ch2_h)

plt.plot(time_revision_r, ch1_r, label='reinforced')
plt.plot(time_revision_h, ch1_h, label='hole')
plt.legend()
plt.xlim(0,0.0003)
plt.show()
