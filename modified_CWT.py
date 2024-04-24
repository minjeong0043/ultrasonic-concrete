import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
from scipy.signal import find_peaks

def cwt(f, t0 = 20):
    '''
    f : input signal
    t0 : minimum length of wavelet
    '''
    f_len = len(f)
    wavelet_width = 500
    result = np.array(list(map(wavelet_convolution, [(f, x) for x in range(t0, f_len, wavelet_width)])))
    #(f,x)인 튜플이 입력값 | wavelet 길이를 지정한 값만큼씩 커지게 함 | wavelet length가 커질수록 저주파 영역을 다루게 된다. 즉, 모든 주파수 영역대를 다루기 위해서 계속 길이를 바꾸면서 수행함.
    return result, wavelet_width

def morlet(T, f0=50):
    '''
	  T : parameter for adjusting length of wavelet
	  f0 : parameter for time-frequenct resolution trade off  || 커질수록 frequency resolution 증가, time resolution 감소
    '''
    x = np.linspace(-2 * np.pi, 2 * np.pi, T)
    return (np.pi ** -0.25) * np.exp(1j * f0 * x - x ** 2 / 2)
def wavelet_convolution(tup):
    f = tup[0]
    T = tup[1]
    f_len = np.shape(f)[0]
    f_hat = np.append(f, np.zeros(T))
    h = morlet(T)
    h_hat = np.append(h, np.zeros(f_len))
    return ifft(fft(f_hat)*fft(h_hat))[round(T/2) : round(T/2) + f_len]


# CWT를 하되, main freq 주변 부위 제거하고 CWT돌리기!

data_path = "C:\\Users\\SM-PC\\Desktop\\초음파신호분석_논문대비\\raw_data\\tp\\r\\240327_tek0006ALL.csv"

data = pd.read_csv(data_path, skiprows=20)
time = data['TIME']
ch1 = data['CH1']

n = len(ch1)
cwt_result, wavelet_length = cwt(ch1)
print(cwt_result)

#normalization
mag = np.abs(cwt_result)
max_value = np.max(mag)

normal_cwt = np.abs(cwt_result)/max_value
print(max_value)
print(np.abs(normal_cwt))

'''
threshold = 10
magnitude = np.abs(cwt_result)
max_value = np.max(magnitude)
exceeds_threshold = magnitude > threshold

cwt_result[exceeds_threshold] = max_value * (cwt_result[exceeds_threshold] / np.abs(cwt_result[exceeds_threshold]))
'''
# 결과 시각화
#plt.imshow(np.abs(cwt_result), extent=[t[0], t[-1], 20, n], aspect='auto',cmap='jet', vmin=0, vmax=20)
plt.imshow(normal_cwt, extent=[time.iloc[0], time.iloc[-1], 20, n], aspect='auto',cmap = 'jet', vmin=0, vmax=0.1)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title(f'Continuous Wavelet Transform(wavelet_width = {wavelet_length})')

plt.show()

