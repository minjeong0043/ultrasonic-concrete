import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 데이터 로드
data_path ="C:\\Users\\SM-PC\\Desktop\\초음파_modified CWT\\초음파_modified CWT\\raw_data\\h_plain\\tp_240325_tek0061ALL.csv"
data = pd.read_csv(data_path, skiprows=20)
time = data['TIME']
ch1 = data['CH1']

# FFT 계산
n = len(ch1)
Ts = (time.iloc[-1] - time.iloc[0]) / len(time)
Fs = 1 / Ts
freq = np.fft.fftfreq(n, d=Ts)[:n // 2]
Y = np.fft.fft(ch1) / n
Y = Y[:n // 2]
Y = np.abs(Y)/np.max(Y)
magnitude = abs(Y)

E = []
for i in range(len(magnitude)):
    E.append(np.power(abs(magnitude[i]),2)/2)
E = np.abs(E)
E = E/np.max(E)
# 피크 찾기
peaks, properties = find_peaks(E, height=0)  # 모든 피크 찾기

# 피크의 진폭을 가져옴
peak_magnitudes = properties['peak_heights']

# 진폭 정렬 및 세 번째로 큰 값 찾기
sorted_peaks = np.argsort(peak_magnitudes)[::-1]  # 진폭을 내림차순으로 정렬
third_largest_peak_index = sorted_peaks[1]  # 네 번째로 큰 값의 인덱스

# 세 번째로 큰 피크 정보 출력
third_largest_peak_freq = freq[peaks[third_largest_peak_index]]
third_largest_peak_magnitude = peak_magnitudes[third_largest_peak_index]
print(f"The third largest peak is at {third_largest_peak_freq} Hz with a magnitude of {third_largest_peak_magnitude}.")

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(freq, E, label='FFT Magnitude')
plt.scatter(freq[peaks], E[peaks], color='red', label='Peaks')
plt.scatter([third_largest_peak_freq], [third_largest_peak_magnitude], color='green', s=100, label='Third Largest Peak')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Energy')
plt.title('PSD and Peaks Highlight')
plt.xlim([0, np.max(freq)])  # 전체 주파수 범위를 보여줌
plt.show()