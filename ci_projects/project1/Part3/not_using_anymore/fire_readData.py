import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

fp = open('forestfires.csv', 'r')
X = np.zeros((517, 11))					#input Data
Y = np.zeros((517, 1))					#labels
row = 0

for line in fp.readlines():
	data = line.strip()
	data = line.split(',')

	x_loc = data[0]
	x_loc = float(x_loc)

	y_loc = data[1]
	y_loc = float(y_loc)

	month = data[2]
	if(month == "jan"):
		month = 1
	elif(month == "feb"):
		month = 2
	elif(month == "mar"):
		month = 3
	elif(month == "apr"):
		month = 4
	elif(month == "may"):
		month = 5
	elif(month == "jun"):
		month = 6
	elif(month == "jul"):
		month = 7
	elif(month == "aug"):
		month = 8
	elif(month == "sep"):
		month = 9
	elif(month == "oct"):
		month = 10
	elif(month == "nov"):
		month = 11
	elif(month == "dec"):
		month = 12


	FFMC = data[3]
	FFMC = float(FFMC)

	DMC = data[4]
	DMC = float(DMC)

	DC = data[5]
	DC = float(DC)

	ISI = data[6]
	ISI = float(ISI)

	temp = data[7]
	temp = float(temp)

	RH = data[8]
	RH = float(RH)

	wind = data[9]
	wind = float(wind)

	rain = data[10]
	rain = float(rain)

	area = data[11]
	area = float(area)

	#X,Y,month,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area

	Y[row][0] = area 
	X[row][0] = x_loc
	X[row][1] = y_loc
	X[row][2] = month
	X[row][3] = FFMC
	X[row][4] = DMC
	X[row][5] = DC
	X[row][6] = ISI
	X[row][7] = temp
	X[row][8] = RH
	X[row][9] = wind
	X[row][10] = rain

	row += 1

fp.close()

print(str(X) + "\n")
print(str(Y))