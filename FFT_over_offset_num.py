
# FFT의 Y 1)offset정하고 그 값보다 큰 값의 개수 /3)freq범위
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def num_index(data_path, offset_Y):
    data = pd.read_csv(data_path, skiprows=20)
    time = data['TIME']
    ch1 = data['CH1']
    ch2 = data['CH2']

    n = len(ch1)
    k = np.arange(n)
    Ts = (time.iloc[-1] - time.iloc[0]) / len(time)
    Fs = 1 / Ts
    T = n / Fs
    freq = k / T
    freq = freq[range(int(n / 2))]

    Y = np.fft.fft(ch1) / n
    Y = Y[range(int(n / 2))]
    Y = abs(Y)
    Y = Y / np.max(Y)

    # 1) offset보다 큰 값의 개수
    indices_over_offset = np.where(Y >= offset_Y)[0]
    indices_over_offset = indices_over_offset[1:]
    # print(indices_over_offset)
    # print(f"0.3넘어가는 index 개수 : {len(indices_over_offset)}")
    num_indices_over_offset = len(indices_over_offset)

    # 3) offset인 값의 Hz 범위
    index_min = indices_over_offset[0]
    index_max = indices_over_offset[-1]
    range_over_range = freq[index_max] - freq[index_min]

    return num_indices_over_offset, range_over_range

    '''
data_path = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240404_ver3\\초음파신호 ToF 분석\\raw_data\\240402\\" \
            "tek0055ALL.csv"
data = pd.read_csv(data_path, skiprows=20)
time = data['TIME']
ch1 = data['CH1']
ch2 = data['CH2']

n = len(ch1)
k = np.arange(n)
Ts = (time.iloc[-1] - time.iloc[0])/len(time)
Fs = 1/ Ts
T = n/Fs
freq = k/T
freq = freq[range(int(n/2))]

Y = np.fft.fft(ch1)/n
Y = Y[range(int(n/2))]
Y = abs(Y)
Y = Y/np.max(Y)

offset_Y = np.max(Y) * 0.3

#1) offset보다 큰 값의 개수
indices_over_offset = np.where(Y>=offset_Y)[0]
indices_over_offset = indices_over_offset[1:]
#print(indices_over_offset)
#print(f"0.3넘어가는 index 개수 : {len(indices_over_offset)}")

#3) offset인 값의 Hz 범위
index_min = indices_over_offset[0]
index_max = indices_over_offset[-1]
print(f" min Hz : {freq[index_min]}")
print(f" max Hz: {freq[index_max]}")
print(f"Hz 범위 : {freq[index_max]-freq[index_min]}")
plt.plot(freq, Y)
plt.scatter(freq[index_min], Y[index_min], color='red')
plt.scatter(freq[index_max], Y[index_max], color='red')
plt.xlim([10000,200000])
plt.show()
'''
