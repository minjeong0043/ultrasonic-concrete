import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def FFT_sumYE_rl(data_path):
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
     Y = np.abs(Y)/np.max(Y) #normalization

     #에너지 계산
     E = []
     for i in range(len(freq)):
          E.append(np.power(abs(Y[i]),2)/2)
     E = E[:len(freq)]

     maxE_index = np.where(E == np.max(E))[0][0]
     #(maxE_index)
     #print(freq[maxE_index])

     #양쪽으로 10k씩~
     min_freq_index = np.where(freq >= (freq[maxE_index]-15000))[0][0]
     max_freq_index = np.where(freq >= (freq[maxE_index]+15000))[0][0]
     #print(min_freq_index)

     # 첫 번째로 : maxE 양쪽으로의 합!
     sum_E_l = np.sum(E[min_freq_index:maxE_index])
     sum_E_r = np.sum(E[maxE_index:max_freq_index])

     #print(f"sum_E_l : {sum_E_l}")
     #print(f"sum_E_r : {sum_E_r}")
     # 두 번째로 : maxE 양쪽으로 처음 peak가 나오는 부분까지의 합!
     peaks, _ = find_peaks(E, height=0)
     max_peak = np.where(peaks== maxE_index)[0][0]

     l_peak_index = peaks[max_peak-1]
     r_peak_index = peaks[max_peak+1]

     sum_E_peak_l = np.sum(E[min_freq_index:l_peak_index])
     sum_E_peak_r = np.sum(E[r_peak_index:max_freq_index])

     #print(f"sum_E_peak_l : {sum_E_peak_l}")
     #print(f"sum_E_peak_r : {sum_E_peak_r}")

     return sum_E_l, sum_E_r, sum_E_peak_l, sum_E_peak_r

'''
# 에너지 측면에서, FFT 측면에서 masY_freq sum과 other freq sum(r,l) 비교

#왼쪽 FFT결과의 합과 오른쪽
# 철근/홀 = FFT 대칭, clear = FFT 비대칭
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:\\Users\\SM-PC\\Desktop\\초음파신호분석_논문대비\\raw_data\\r_clear\\240402_tek0050ALL.csv"

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
Y = np.abs(Y)/np.max(Y) #normalization

#에너지 계산
E = []
for i in range(len(freq)):
     E.append(np.power(abs(Y[i]),2)/2)
E = E[:len(freq)]

maxE_index = np.where(E == np.max(E))[0][0]
#(maxE_index)
#print(freq[maxE_index])

#양쪽으로 10k씩~
min_freq_index = np.where(freq >= (freq[maxE_index]-15000))[0][0]
max_freq_index = np.where(freq >= (freq[maxE_index]+15000))[0][0]
#print(min_freq_index)

# 첫 번째로 : maxE 양쪽으로의 합!
sum_E_l = np.sum(E[min_freq_index:maxE_index])
sum_E_r = np.sum(E[maxE_index:max_freq_index])

print(f"sum_E_l : {sum_E_l}")
print(f"sum_E_r : {sum_E_r}")
# 두 번째로 : maxE 양쪽으로 처음 peak가 나오는 부분까지의 합!
peaks, _ = find_peaks(E, height=0)
max_peak = np.where(peaks== maxE_index)[0][0]

l_peak_index = peaks[max_peak-1]
r_peak_index = peaks[max_peak+1]

sum_E_peak_l = np.sum(E[min_freq_index:l_peak_index])
sum_E_peak_r = np.sum(E[r_peak_index:max_freq_index])

print(f"sum_E_peak_l : {sum_E_peak_l}")
print(f"sum_E_peak_r : {sum_E_peak_r}")

plt.plot(freq, E)
plt.scatter(freq[maxE_index], E[maxE_index], color='red')
plt.scatter(freq[l_peak_index], E[l_peak_index], color='blue')
plt.scatter(freq[r_peak_index], E[r_peak_index], color='blue')
plt.xlim([30000, 70000])
plt.show()

'''




