
import pandas as pd
import numpy as np

def calculate_tof_vel(data_path):
    data = pd.read_csv(data_path, skiprows=20)

    time = pd.to_numeric(data['TIME'], errors='coerce').to_numpy()
    ch1 = pd.to_numeric(data['CH1'], errors='coerce').to_numpy()
    ch2 = pd.to_numeric(data['CH2'], errors='coerce').to_numpy()

    # 신호 없을 때의 오프셋 값 찾기
    offset_ch1 = ch1[:100]
    offset_time_ch1 = time[:100]

    mag_max_ch1 = np.max(offset_ch1)
    mag_min_ch1 = np.min(offset_ch1)
    print(mag_max_ch1)
    if mag_max_ch1 >= 0:
        mag_max_ch1 = round(mag_max_ch1, 3)
    else:
        mag_max_ch1 = -round(-mag_max_ch1, 3)

    if mag_min_ch1 >= 0:
        mag_min_ch1 = round(mag_min_ch1, 3)
    else:
        mag_min_ch1 = -round(-mag_min_ch1, 3)

    # ch1에 대해서는 mag가 신호가 들어온 곳의 max값 min 값의 범위를 넘어가면 신호가 들어온 것으로 판단.
   # indices_ch1 = np.where((ch1 > (mag_max_ch1 + 0.001)) | (ch1 < (mag_min_ch1 - 0.001)))[0]
    indices_ch1 = np.where((ch1 > (mag_max_ch1 + 0.003)) | (ch1 < (mag_min_ch1 - 0.003)))[0]
    index_ch1 = indices_ch1[0]

    # ch2 offset
    #offset_ch2 = ch2[:50]
    #offset_time_ch2 = time[:50]
    offset_ch2 = ch2[:70]
    offset_time_ch2 = time[:70]

    mag_max_ch2 = np.max(offset_ch2)
    mag_min_ch2 = np.min(offset_ch2)

    if mag_max_ch2 >= 0:
        mag_max_ch2 = round(mag_max_ch2, 2)
    else:
        mag_max_ch2 = -round(-mag_max_ch2, 2)

    if mag_min_ch2 >= 0:
        mag_min_ch2 = round(mag_min_ch2, 2)
    else:
        mag_min_ch2 = -round(-mag_min_ch2, 2)

    # ch2에 대해서는 mag가 0이상이 되면 신호가 들어온 것으로 판단.
    # threshold_ch2 = 0
    # indices_ch2 = np.where(ch2>0)[0]

    #indices_ch2 = np.where((ch2 > (mag_max_ch2 + 0.01)) | (ch2 < (mag_min_ch2 - 0.01)))[0]
    indices_ch2 = np.where((ch2 > (mag_max_ch2 + 0.05)) | (ch2 < (mag_min_ch2 - 0.03)))[0]
    index_ch2 = indices_ch2[0] - 1
    print(f"arrival time : {time[index_ch1]}")
    print(f"start time : {time[index_ch2]}")
    tof = -time[index_ch2] + time[index_ch1]
    vel = (20/100)/tof
    return tof, vel


# 파일 하나 확인 코드
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석\\raw_data\\240402\\tek0165ALL.csv"
data = pd.read_csv(data_path, skiprows=20)

time = data['TIME']
ch1 = data['CH1']
ch2 = data['CH2']

# 신호 없을 때의 오프셋 값 찾기
offset_ch1 = ch1[:200]
offset_time_ch1 = time[:200]

mag_max_ch1 = np.max(offset_ch1)
mag_min_ch1 = np.min(offset_ch1)

if mag_max_ch1>=0:
    mag_max_ch1 = round(mag_max_ch1,3)
else:
    mag_max_ch1 = -round(-mag_max_ch1,3)

if mag_min_ch1>=0:
    mag_min_ch1 = round(mag_min_ch1,3)
else:
    mag_min_ch1 = -round(-mag_min_ch1,3)

# ch1에 대해서는 mag가 신호가 들어온 곳의 max값 min 값의 범위를 넘어가면 신호가 들어온 것으로 판단.
indices_ch1 = np.where((ch1>(mag_max_ch1+0.002))|(ch1<(mag_min_ch1-0.002)))[0]
index_ch1 = indices_ch1[0]


# ch2 offset
offset_ch2 = ch2[:50]
offset_time_ch2 = time[:50]

mag_max_ch2 = np.max(offset_ch2)
mag_min_ch2 = np.min(offset_ch2)

if mag_max_ch2>=0:
    mag_max_ch2 = round(mag_max_ch2,2)
else:
    mag_max_ch2 = -round(-mag_max_ch2,2)

if mag_min_ch2>=0:
    mag_min_ch2 = round(mag_min_ch2,2)
else:
    mag_min_ch2 = -round(-mag_min_ch2,2)

# ch2에 대해서는 mag가 0이상이 되면 신호가 들어온 것으로 판단.
#threshold_ch2 = 0
#indices_ch2 = np.where(ch2>0)[0]

indices_ch2 = np.where((ch2>(mag_max_ch2+0.05))|(ch2<(mag_min_ch2-0.01)))[0]
index_ch2 = indices_ch2[0]-1
print(f"arrival time : {time[index_ch1]}")
print(f"start time : {time[index_ch2]}")
tof = -time[index_ch2] + time[index_ch1]

vel = (20/100)/tof
print(f"tof : {tof}")
print(f"vel : {vel}")
plt.plot(time, ch1)
plt.plot(time, ch2)
plt.scatter(time.iloc[index_ch1], ch1[index_ch1], color = 'red', label = 'arrival time')
plt.scatter(time.iloc[index_ch2], ch2[index_ch2], color = 'blue', label = 'start time')
plt.legend()
plt.show()

'''
