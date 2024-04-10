import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft


# one side spectrum
def one_side(padded_t, X):
    Ts = padded_t[1]-padded_t[0]
    Fs = 1.0/Ts
    N = len(X)
    n = np.arange(N)
    T = N/Fs
    freq = n/T

    n_oneside = N//2
    f_oneside = freq[:n_oneside]
    X_oneside = X[:n_oneside]/n_oneside

    return f_oneside, X_oneside

# PSD and cumulative PSD
def PSD(X_oneside):
    E = []
    for i in range(len(X_oneside)):
        E.append(np.power(abs(X_oneside[i]),2)/2)

    return E[:len(X_oneside)]

# fmin 20000, fmax 100000
def cut_frequency(f_oneside,E):
    E_cut = []
    F_cut = []
    for i in range(1, len(f_oneside)):
        if f_oneside[i] > 100000:
            fmax = i
            break

    for i in range(1, len(f_oneside)):
        if f_oneside[i] > 20000:
            fmin = i
            break

    for i in range(fmin, fmax):
        E_cut.append(E[i])
        F_cut.append(f_oneside[i])

    return F_cut, E_cut

#'rth' moment of PSD curve
def moment(r,N, E, f_oneside):
    #N = len(E)
    m = []
    f = []
    total = 0
    del_f = f_oneside[1]-f_oneside[0]
    for i in range(1, int(N/2)):
        total += E[i] * np.power(f_oneside[i], r) * del_f
        m.append(total)
        f.append(f_oneside[i])
    return f, m


def data_to_list(data_name):
    data = pd.read_csv(data_name, skiprows=20)

    x_data = data['CH1']
    t_data = data['TIME']
    return t_data, x_data

def draw_rawdata(data_name, save_path=None):
    t_data, x_data = data_to_list(data_name)
    plt.figure(figsize=(15,8))
    plt.plot(t_data, x_data)
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Raw Data')

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def draw_PSD(data_name, save_path=None):
    data = pd.read_csv(data_name, skiprows=20)
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
    Y = Y/np.max(Y)
    E = PSD(Y)
    F_cut, E_cut = cut_frequency(freq, E)
    c_f, c_m = moment(0, 2 * len(E_cut), E_cut, F_cut)
    f_1, m_1 = moment(1, len(E_cut), E_cut, F_cut)
    f_0, m_0 = moment(0, len(E_cut), E_cut, F_cut)
    f_3, m_3 = moment(3, len(E_cut), E_cut, F_cut)
    CF = m_1[len(m_1) - 1] / m_0[len(m_1) - 1]
    TM = m_3[len(m_3) - 1]
    print("CF: ", CF)
    print("TM: ", TM / 10000000000000)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(time, ch1)
    ax[0].set_xlabel("time(sec)")
    ax[0].set_ylabel("amp(V)")
    ax[0].set_title("Time domain")

    ax[1].plot(F_cut, E_cut)
    ax[1].set_xlabel("freq(Hz)")
    ax[1].set_ylabel("Mag(V^2)")
    ax[1].set_xlim(20000, 100000)
    #ax[1].set_ylim(-0.0001, 0.003)
    ax[1].set_title("PSD")

    ax[2].plot(c_f, c_m)
    ax[2].set_xlabel("freq(Hz)")
    ax[2].set_ylabel("Mag(V^2*Hz)")
    ax[2].set_xlim(0, 100000)
    #ax[2].set_ylim(-0.5, 10)
    ax[2].set_title("cumulative PSD")
    plt.annotate(f'CF: {CF}\nTM : {TM / 10000000000000}\nMax_PSD : {np.max(E_cut)}\ntotle_E : {np.max(c_m)}', xy=(3, 10), xytext=(4, 6),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.subplots_adjust(hspace=0.8)

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def draw_FFT(data_name, save_path=None):
    t_data, x_data = data_to_list(data_name)
    n = len(x_data)
    k = np.arange(n)
    #Ts = t_data[32]-t_data[31] # 너무 작은 값으로 나와서 Fs 구할 때 나누기가 안 됨. 따라서 특정 값으로 넣어주었음.
    Ts = (t_data.iloc[-1]-t_data.iloc[0])/len(t_data)
    Fs = 1/Ts
    T = n/Fs
    freq = k/T
    freq = freq[range(int(n/2))]

    Y = np.fft.fft(x_data)/n
    Y = Y[range(int(n/2))]
    #Y = Y/max(abs(Y)) #normalization

    plt.plot(freq, abs(Y))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Mag')
    plt.title('FFT')
    plt.xlim([0, 200000])
    plt.ylim([-0.01, 0.1])

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def draw_STFT(data_name, save_path=None):
    t_data, x_data = data_to_list(data_name)
    Ts = 40e-9
    Fs = 1 / Ts
    win_length = 250 * 25
    [f, t, Zxx] = signal.stft(x_data, Fs, nperseg=win_length)

    v_max = np.log(np.max(np.abs(Zxx)) ** 2)
    v_min = np.log(np.min(np.abs(Zxx)) ** 2) + 17

    ymin = 10e3
    ymax = 180e3

    ## plotting STFT color map

    plt.figure(figsize=(12, 5))

    plt.pcolormesh(t, f, np.log(np.abs(Zxx) ** 2), vmin=v_min, vmax=v_max, shading='auto')  # 'flat _ dimension error

    plt.title('STFT Magnitude')
    plt.xlabel('Time [sec]')
    plt.ylim([ymin, ymax])
    plt.ylabel('Frequency [Hz]')

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def draw_CWT(data_name, save_path=None):
    t_data, x_data = data_to_list(data_name)

    n = len(x_data)
    cwt_result = cwt(x_data)

    t = []
    for i in range(len(t_data)):
        t.append(t_data[i])
    # 결과 시각화
    plt.imshow(np.abs(cwt_result), extent=[t[0], t[-1], 20, n], aspect='auto',
               cmap='jet')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Continuous Wavelet Transform(wavelet_width = 1000)')

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()
# function for cwt
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

def cwt(f, t0 = 20):
    '''
    f : input signal
    t0 : minimum length of wavelet
    '''
    f_len = len(f)
    wavelet_width = 1000
    result = np.array(list(map(wavelet_convolution, [(f, x) for x in range(t0, f_len, wavelet_width)])))
    #(f,x)인 튜플이 입력값 | wavelet 길이를 지정한 값만큼씩 커지게 함 | wavelet length가 커질수록 저주파 영역을 다루게 된다. 즉, 모든 주파수 영역대를 다루기 위해서 계속 길이를 바꾸면서 수행함.
    return result

def save_data(input_directory, output_directory, type =None):
    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            # CSV 파일 경로 생성
            csv_file_path = os.path.join(input_directory, filename)

            # 함수 호출 및 이미지 저장
            if type == 'FFT':
                # 이미지 파일 경로 생성
                image_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_FFT.png")
                draw_FFT(csv_file_path, save_path=image_file_path)
            elif type == 'STFT':
                image_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_STFT.png")
                draw_STFT(csv_file_path, save_path=image_file_path)
            elif type == 'CWT':
                image_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_CWT.png")
                draw_CWT(csv_file_path, save_path=image_file_path)
            elif type == 'RAW':
                image_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_RAW.png")
                draw_rawdata(csv_file_path, save_path=image_file_path)
            elif type == 'PSD':
                if not os.path.exists(output_directory):
                    os.mkdir(output_directory)
                image_file_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_PSD.png")
                draw_PSD(csv_file_path, save_path=image_file_path)

    print("Conversion complete.")
