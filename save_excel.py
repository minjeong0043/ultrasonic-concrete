import glob
import os
from openpyxl import Workbook
from record_module import calculate_tof_vel, FFT_max
folder_name = '240402'
input_folder = os.path.join("C:\\Users\\SM-PC\\Desktop\\초음파신호 ToF 분석_240404_ver2\\초음파신호 ToF 분석\\raw_data", folder_name)
output_folder = f"C:\\Users\\SM-PC\\Desktop\\초음파신호 ToF 분석_240404_ver2\\초음파신호 ToF 분석\\"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

save_path = os.path.join(output_folder, folder_name+ ".xlsx")

wb = Workbook()
ws = wb.active0
ws.title = "FFT left,right sum"
ws.append(['File Name', 'tof', 'vel'])

for file_path in glob.glob(os.path.join(input_folder, '*.csv')):
    file_name = os.path.basename(file_path)
    tof, vel = calculate_tof_vel(file_path)
    ws.append([file_name, tof, vel])

wb.save(save_path)
