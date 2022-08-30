
import can
import csv

filename = "TSMaster_2022_05_20_16_22_46.blf"
log = can.BLFReader(filename)
log = list(log)

log_output = []

for msg in log:
    msg = str(msg)
    msg=msg.split(' ')
    print(msg)
    log_output.append([msg[1],msg[10],msg[15],msg[36:44],msg[49]])

with open("../output2.csv", "w", newline='') as f:
    writer = csv.writer(f,delimiter=';', quotechar='\"', quoting=csv.QUOTE_ALL)
    writer.writerows(log_output)