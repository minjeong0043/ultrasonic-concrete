import pandas as pd
import numpy as np

def FFT_sumYE_rl(data_path):
    data = pd.read_csv(data_path, skiprows=20)
    time = data['TIME']
    ch1 = data['CH1']

    n = len(ch1)
    Ts = (time.iloc[-1] - time.iloc[0]) / len(time)
    Fs = 1 / Ts
    T = n / Fs
    freq = np.arange(n) / T
    freq = freq[range(int(n / 2))]

    Y = np.fft.fft(ch1) / n
    Y = Y[range(int(n / 2))]
    Y = np.abs(Y)

    # 에너지 계산
    E = []
    for i in range(len(freq)):
        E.append(np.power(abs(Y[i]), 2) / 2)
    E = E[:len(freq)]

    maxY_index = np.argmax(Y)
    maxY_freq = freq[maxY_index]

    th_freq = 1000
    maxY_freq_range_index = np.where((freq > maxY_freq - th_freq) & (freq < maxY_freq + th_freq))

    main_freq = 54 * 1000
    sumY_l = np.sum(Y[10:(maxY_freq_range_index[0][0] - 1)])
    sumY_r = np.sum(Y[(maxY_freq_range_index[0][-1] + 1):np.argwhere(freq >= main_freq * 3)[0][0]])
    sumY_all = np.sum(Y)

    sumE_l = np.sum(E[10:(maxY_freq_range_index[0][0] - 1)])
    sumE_r = np.sum(E[(maxY_freq_range_index[0][-1] + 1):np.argwhere(freq >= main_freq * 3)[0][0]])
    sumE_all = np.sum(E)
    mainE_sum = np.sum(E[maxY_freq_range_index[0][0]:maxY_freq_range_index[0][-1]])
    print(f"sumY_all : {sumY_all}")
    print(f"sum_r : {sumY_r}")
    print(f"sum_l : {sumY_l}")
    print(f"ratio : {(sumY_l+sumY_r)/sumY_all}")

    return mainE_sum, (sumE_l+sumE_r)/mainE_sum

'''
# 에너지 측면에서, FFT 측면에서 masY_freq sum과 other freq sum(r,l) 비교

#왼쪽 FFT결과의 합과 오른쪽
# 철근/홀 = FFT 대칭, clear = FFT 비대칭
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석\\raw_data\\r\\240403_tek0008ALL.csv"

data = pd.read_csv(data_path, skiprows=20)
time = data['TIME']
ch1 = data['CH1']

n = len(ch1)
Ts = (time.iloc[-1]-time.iloc[0])/len(time)
Fs = 1/Ts
T = n/Fs
freq = np.arange(n)/T
freq = freq[range(int(n/2))]

Y = np.fft.fft(ch1)/n
Y = Y[range(int(n/2))]
Y = np.abs(Y)

#에너지 계산
E = []
for i in range(len(freq)):
     E.append(np.power(abs(Y[i]),2)/2)
E = E[:len(freq)]


maxY_index = np.argmax(Y)
maxY_freq = freq[maxY_index]

#maxY_freq기준 += 1000를 범위로 잡고 그 부분과 아닌 부분 나누기
#maxY
th_freq = 1000
maxY_freq_range_index = np.where((freq>maxY_freq - th_freq)&(freq<maxY_freq + th_freq))

#print(maxY_freq_range_index[0][-1])

#왼쪽 오른쪽의 freq 합
main_freq = 54*1000
sumY_l = np.sum(Y[10:(maxY_freq_range_index[0][0]-1)])
sumY_r = np.sum(Y[(maxY_freq_range_index[0][-1]+1):np.argwhere(freq >= main_freq*3)[0][0]])
sumY_all = np.sum(Y)
print(f"sumY_all : {sumY_all}")
print(f"sum_r : {sumY_r}")
print(f"sum_l : {sumY_l}")
print(f"ratio : {(sumY_)}")
#에너지 측면에서
main_freq = 54*1000
sumE_l = np.sum(E[10:(maxY_freq_range_index[0][0]-1)])
sumE_r = np.sum(E[(maxY_freq_range_index[0][-1]+1):np.argwhere(freq >= main_freq*3)[0][0]])
print(f"sum_r : {sumE_r}")
print(f"sum_l : {sumE_l}")

plt.subplot(2,1,1)
plt.plot(freq, Y)
plt.scatter(maxY_freq, Y[maxY_index], color='red')
plt.plot(freq[maxY_freq_range_index], Y[maxY_freq_range_index])
plt.xlim([-1000, 200000])
plt.subplot(2,1,2)
plt.plot(freq, E)
plt.xlim([-1000, 200000])
plt.show()



'''