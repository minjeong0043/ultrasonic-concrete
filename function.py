from module import draw_PSD, draw_rawdata, draw_FFT, draw_STFT, draw_CWT, save_data

#data_name = 'C:\\Users\\SM-PC\\Desktop\\소테리아_실험데이터\\1005\\tek0000ALL.csv'

# 하나의 파일 이미지 확인
#draw_rawdata(data_name)

#draw_FFT(data_name)

#draw_STFT(data_name)

#draw_CWT(data_name)

# 폴더 내부 데이터 한 번에 이미지 변환
input_directory = 'C:\\Users\\김민정\\Desktop\\초음파_modified CWT\\raw_data\\r_plain'
output_directory = "C:\\Users\\김민정\\Desktop\\초음파_modified CWT\\img\\CWT_r_plain"

#type에 RAW, FFT, STFT, CWT, CWT_th 입력
save_data(input_directory, output_directory, type='CWT_th')
