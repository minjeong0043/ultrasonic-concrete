from soteria_code_module_fin import draw_PSD, draw_rawdata, draw_FFT, draw_STFT, draw_CWT, save_data

#data_name = 'C:\\Users\\SM-PC\\Desktop\\소테리아_실험데이터\\1005\\tek0000ALL.csv'

# 하나의 파일 이미지 확인
#draw_rawdata(data_name)

#draw_FFT(data_name)

#draw_STFT(data_name)

#draw_CWT(data_name)

# 폴더 내부 데이터 한 번에 이미지 변환
input_directory = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석\\raw_data\\240405(2)"
output_directory = "C:\\Users\\김민정\\Desktop\\초음파신호 ToF 분석_240405_ver4\\초음파신호 ToF 분석\\img_raw\\total_normal_PSD_240405(2)"

#type에 RAW, FFT, STFT, CWT 입력
save_data(input_directory, output_directory, type='PSD')

