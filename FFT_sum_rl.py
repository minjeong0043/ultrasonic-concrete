#왼쪽 FFT결과의 합과 오른쪽
# 철근/홀 = FFT 대칭, clear = FFT 비대칭
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def sum_fft_rl(data_path):
    #data_path = "C:\\Users\\SM-PC\\Desktop\\초음파신호 ToF 분석_240404_ver2\\초음파신호 ToF 분석\\raw_data\\240402\\tek0000ALL.csv"
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
    # 0 부터 CF-1 까지의 배열/ CF+1부터 CF+1+CF까지의 배열
    Max_mag = np.max(Y)
    index_max = np.where(Y == Max_mag)[0][0]
    array_1 = Y[9:index_max+1]
    array_1 = array_1[::-1]
    array_2 = Y[index_max:]

    sub = []
    for i in range(len(array_1)):
        sub.append(array_1[i] - array_2[i])

    array_2 = array_2[:len(array_1)]
    sum_left = np.sum(array_1)
    sum_right = np.sum(array_2)
    sum_sub = np.sum(sub)

    '''
    print(sum_left)
    print(sum_right)
    print(sum_sub)
    plt.subplot(3,1,1)
    plt.plot(freq[9:(index_max+1)], array_1)
    plt.title('left')
    plt.subplot(3,1,2)
    plt.plot(freq[9:index_max+1], array_2)
    plt.title('right')
    plt.subplot(3,1,3)
    plt.plot(freq[9:index_max+1], np.abs(sub))
    plt.title('abs(sub)')
    plt.subplots_adjust(hspace=0.8)
    plt.show()'''
    return sum_left, sum_right, sum_sub