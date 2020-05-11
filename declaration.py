#! /anaconda3/envs/Flasher/bin/python

# Time scan:
# scan_type : 'time' (surface plots and stability for run pixel by pixel)

# Surface scan files:
# scan_type : 'space' (surface plots and stability for run pixel by pixel)
# '2019_04_24/Scan_Flasher01.txt' excluded since it is darkcounts
file_2019_04_24 = ['2019_04_24/Scan_Flasher02.txt',
                   '2019_04_24/Scan_Flasher03.txt',
                   '2019_04_24/Scan_Flasher04.txt',
                   '2019_04_24/Scan_Flasher05.txt',
                   '2019_04_24/Scan_Flasher06.txt']

# Surface scan files:
# scan_type : 'space' (surface plots and stability for run pixel by pixel)
# '2019_04_25/Scan_Flasher00.txt' excluded since it is darkcounts
file_2019_04_25 = ['2019_04_25/Scan_Flasher01.txt',
                   '2019_04_25/Scan_Flasher02.txt',
                   '2019_04_25/Scan_Flasher03.txt',
                   '2019_04_25/Scan_Flasher04.txt',
                   '2019_04_25/Scan_Flasher05.txt',
                   '2019_04_25/Scan_Flasher06.txt',
                   '2019_04_25/Scan_Flasher07.txt',
                   '2019_04_25/Scan_Flasher08.txt',
                   '2019_04_25/Scan_Flasher09.txt',
                   '2019_04_25/Scan_Flasher10.txt',
                   '2019_04_25/Scan_Flasher11.txt',
                   '2019_04_25/Scan_Flasher12.txt']

# Surface scan files:
# scan_type : 'space' (surface plots and stability for run pixel by pixel)
file_2019_04_27 = ['2019_04_27/Scan_Flasher00.txt',
                   '2019_04_27/Scan_Flasher01.txt',
                   '2019_04_27/Scan_Flasher02.txt',
                   '2019_04_27/Scan_Flasher03.txt',
                   '2019_04_27/Scan_Flasher04.txt',
                   '2019_04_27/Scan_Flasher05.txt',
                   '2019_04_27/Scan_Flasher06.txt',
                   '2019_04_27/Scan_Flasher07.txt',
                   '2019_04_27/Scan_Flasher08.txt',
                   '2019_04_27/Scan_Flasher09.txt',
                   '2019_04_27/Scan_Flasher10.txt']

# Thermal variation files:
# scan_type : 'temp' for ClimPilot and 'time' for Scan_Flasher
file_2019_05_03 = ['2019_05_03/ClimPilot00', '2019_05_03/Scan_Flasher00',
                   '2019_05_03/ClimPilot01', '2019_05_03/Scan_Flasher01']

# DATA :
file_list1 = file_2019_04_24
file_list2 = file_2019_04_25
file_list3 = file_2019_04_27
file_list4 = file_2019_05_03

space_Data = [file_list1, file_list2, file_list3]



# OPTIONS :
steps = 11
x_limits = y_limits = [0., 300.]
number_of_cells = steps * steps
