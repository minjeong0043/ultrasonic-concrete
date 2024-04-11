import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# FFT main freq 주변의 합과 그 외 합 가시화
def sum_main_other_freq(data_path):
    data = pd.read_csv(data_path, skiprows=20)
    time = data["TIME"]
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
    Y = np.abs(Y)

    # 저주파 노이즈 제거
    Y = Y[5:]
    freq = freq[5:]

    # 주파수 범위
    main_freq_min = 53000
    main_freq_max = 55000

    indices = np.where((freq >= main_freq_min) & (freq <= main_freq_max))
    print(indices)

    main_freq_sum = np.sum(Y[(freq >= main_freq_min) & (freq <= main_freq_max)])
    other_freq_sum = np.sum(Y[(freq < main_freq_min) | (freq > main_freq_max)])
    print(f'Main Frequency Sum: {main_freq_sum}')
    print(f'Other Frequency Sum: {other_freq_sum}')
    return main_freq_sum, other_freq_sum

'''
data_path = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석\\raw_data\\240402\\tek0000ALL.csv"
data =pd.read_csv(data_path, skiprows=20)
time= data["TIME"]
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
Y = np.abs(Y)

# 저주파 노이즈 제거
Y = Y[5:]
freq = freq[5:]

# 주파수 범위
main_freq_min = 53000
main_freq_max = 55000

indices = np.where((freq>=main_freq_min)&(freq<=main_freq_max))
print(indices)

main_freq_sum = np.sum(Y[(freq>=main_freq_min)&(freq<=main_freq_max)])
other_freq_sum = np.sum(Y[(freq<main_freq_min)|(freq>main_freq_max)])
print(f'Main Frequency Sum: {main_freq_sum}')
print(f'Other Frequency Sum: {other_freq_sum}')

plt.plot(freq, Y)
plt.plot(freq[indices], Y[indices])
plt.show()
'''