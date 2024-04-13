# max peak 기준 양쪽의 peak 개수! (th 이상)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def peaks_num_rl(data_path, noise_threshold):
    data = pd.read_csv(data_path, skiprows=20)

    time = data['TIME']
    ch1 = data['CH1']

    n = len(ch1)
    k = np.arange(n)
    Ts = (time.iloc[-1]-time.iloc[0])/len(time)
    Fs = 1/Ts
    T = n / Fs
    freq = k/T
    freq = freq[range(int(n/2))]

    Y = np.fft.fft(ch1) / n
    Y = Y[range(int(n / 2))]
    Y = np.abs(Y)
    Y = Y / np.max(Y)

    total_peaks_index_threshold, _ = find_peaks(Y, height=noise_threshold)  # noise level 이상으로 height 설정
    total_peaks_index = find_peaks(Y)
    maxY_index = np.argmax(Y)

    # maxY 왼 쪽 peak
    l_peaks_index = total_peaks_index_threshold[total_peaks_index_threshold < maxY_index]

    # maxY 오른 쪽 peak
    r_peaks_index = total_peaks_index_threshold[total_peaks_index_threshold > maxY_index]

    num_l_peaks_index = len(l_peaks_index)
    num_r_peaks_index = len(r_peaks_index)
    num_total_peaks_index = len(total_peaks_index[0])

    return num_l_peaks_index, num_r_peaks_index, num_total_peaks_index


'''
data_path = "C:\\Users\\김민정\\Desktop\\초음파 논문대비 데이터 분석\\raw_data\\r\\240402_tek0000ALL.csv"
data = pd.read_csv(data_path, skiprows=20)

time= data['TIME']
ch1 = data['CH1']

n = len(ch1)
k = np.arange(n)
Ts = (time.iloc[-1] - time.iloc[0])/len(time)
Fs = 1/ Ts
T = n/Fs
freq = k/T
freq = freq[range(int(n/2))]

Y = np.fft.fft(ch1)/n
Y = Y[range(int(n/2))]
Y = np.abs(Y)
Y = Y/np.max(Y)

noise_threshold = 0.1
total_peaks_index, _ =find_peaks(Y, height=noise_threshold) # noise level 이상으로 height 설정

maxY_index = np.argmax(Y)
print(maxY_index)
print(len(total_peaks_index))
# maxY 왼 쪽 peak
l_peaks_index = total_peaks_index[total_peaks_index<maxY_index]
print(len(l_peaks_index))
# maxY 오른 쪽 peak
r_peaks_index = total_peaks_index[total_peaks_index>maxY_index]
print(len(r_peaks_index))


plt.plot(freq, Y)
plt.scatter(freq[r_peaks_index], Y[r_peaks_index], color='red')
plt.scatter(freq[l_peaks_index], Y[l_peaks_index], color='blue')
plt.show()
'''