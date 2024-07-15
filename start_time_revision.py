import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# transducer가 신호 보낼 때를 0으로 시간 축 값 변경하기

data =  "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석_240411_ver6\\초음파신호 ToF 분석\\raw_data\\ap\\r\\240402_tek0000ALL.csv"

data = pd.read_csv(data, skiprows=20)
time = data['TIME']
t = data['CH2']
r = data['CH1']


#transducer에서 시작하는 부분 찾기
# 신호 안 들어왔을 때  max min
no_signal = t[:100]
max = np.max(no_signal)
min = np.min(no_signal)

if max >=0:
    max = round(max, 3)
else:
    max = -round(-max, 3)

if min >=0:
    min = round(min,3)
else:
    min = -round(-min, 3)

c = 0.0005
indices = np.where((t> max+c)|(t<min-c))
print(indices)
index = indices[0][0]
print(index)

#plt.plot(time ,t)
#plt. scatter(time[index], t[index])
#plt.show()


start_time = time[index]
print(start_time)

# start_time 기준으로

print(len(time))
time_revision = [x-start_time for x in time]
print(time[0])
print(time_revision[0])
print(time_revision[0]-time[0])
print(start_time)

plt.subplot(4,1,1)
plt.plot(time, t)
plt.subplot(4,1,2)
plt.plot(time_revision, t)
plt.subplot(4,1,3)
plt.plot(time, r)
plt.subplot(4,1,4)
plt.plot(time_revision, r)
plt.show()