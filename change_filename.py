import pandas as pd
import numpy as np
import os
import glob

# 대상 폴더 지정
folder_path = ('C:\\Users\\SM-PC\\Desktop\\초음파신호 ToF 분석_240410_ver5\\초음파신호 ToF 분석\\raw_data\\'
               '240405(2)')

# 폴더 이름 추출
folder_name = os.path.basename(folder_path)
#folder_name = '240328'

# 폴더 내 모든 .csv 파일 리스트 생성
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

for file_path in csv_files:
    # 현재 파일의 이름과 경로 분리
    directory, filename = os.path.split(file_path)

    # 새 파일명 생성: 폴더명_원래파일명.csv
    new_filename = f"{folder_name}_{filename}"

    # 새 파일 경로 생성
    new_file_path = os.path.join(directory, new_filename)

    # 파일 이름 변경
    os.rename(file_path, new_file_path)
    print(f"{filename} -> {new_filename}")

print("모든 파일 이름 변경 완료.")