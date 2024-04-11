import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#FFT의 max 값과 그의 freq

def max_freq(data_path):
    data = pd.read_csv(data_path, skiprows=20)
    time = data['TIME']
    ch1 = data['CH1']
    ch2 = data['CH2']

    n = len(ch1)
    k = np.arange(n)

    Ts = (time.iloc[-1] - time.iloc[0])/len(time)
    Fs = 1/Ts
    T = n/Fs
    freq = k/T
    freq = freq[range(int(n/2))]

    Y = np.fft.fft(ch1)/n
    Y = Y[range(int(n/2))]

    Y = np.abs(Y)
    max_Y = np.max(np.abs(Y[10:]))

    index = np.where((Y == max_Y)| (Y == -max_Y))[0]

    max_Y_freq = freq[index]

    return max_Y, max_Y_freq
'''
data_path = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석\\raw_data\\240402\\tek0000ALL.csv"
data = pd.read_csv(data_path, skiprows=20)
time = data['TIME']
ch1 = data['CH1']
ch2 = data['CH2']

n = len(ch1)
k = np.arange(n)

Ts = (time.iloc[-1]-time.iloc[0])/len(time)
Fs = 1/Ts
T = n/Fs
freq = k/T
freq = freq[range(int(n/2))]

Y = np.fft.fft(ch1)/n
Y = Y[range(int(n/2))]

abs_Y = np.abs(Y)
max_Y = np.max(np.abs(abs_Y[10:]))

index = np.where((abs_Y == max_Y) |(abs_Y == -max_Y))[0]

print(freq[index])
print(abs_Y[index])

plt.subplot(2,1,1)
plt.plot(time, ch1)
plt.subplot(2,1,2)
plt.plot(freq, abs(Y))
plt.xlim([0,200000])
plt.scatter(freq[index], abs(Y[index]), color='red')
plt.show()
'''