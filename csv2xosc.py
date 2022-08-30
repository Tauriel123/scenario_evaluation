from ektOpens import xosc_write_V3
import os


def csv2xosc(rootPath):
    for root, dirs, files in os.walk(rootPath):
        for file in files:
            csv_path = rootPath+file
            output_path = 'C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\xosc_for_scenario\\' + csv_path[
                -5] + '.xosc'
             # print(nds_path)
            xosc_write_V3(csv_path, '', output_path)


rootPath = 'C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\csv_for_scenario\\'
csv2xosc(rootPath)
# nds_path='C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\example.xlsx'
# nds_path='C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\example1.csv'
# output_path='C:\\Users\\tauriel\\OneDrive\\桌面\\寒假研究进展\\NGSIM数据\\example1.xosc'
# xosc_write_V3(nds_path, '', output_path)