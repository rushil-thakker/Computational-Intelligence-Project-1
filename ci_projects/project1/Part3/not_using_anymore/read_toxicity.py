import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

fp = open('toxicity.csv', 'r')
X = np.zeros((546, 8))					#input Data
Y = np.zeros((546, 1))					#labels
row = 0

for line in fp.readlines():
	data = line.strip()
	data = line.split(';')

	TPSA = data[0]
	TPSA = float(TPSA)

	SAacc = data[1]
	SAacc = float(SAacc)

	H050 = data[2]
	H050 = float(H050)

	MLOGP = data[3]
	MLOGP = float(MLOGP)

	RDCHI = data[4]
	RDCHI = float(RDCHI)

	GATS1p = data[5]
	GATS1p = float(GATS1p)

	nN = data[6]
	nN = float(nN)

	C040 = data[7]
	C040 = float(C040)

	LC50 = data[8]
	LC50 = float(LC50)

	#X,Y,month,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area

	Y[row][0] = LC50 
	X[row][0] = TPSA
	X[row][1] = SAacc
	X[row][2] = H050
	X[row][3] = MLOGP
	X[row][4] = RDCHI
	X[row][5] = GATS1p
	X[row][6] = nN
	X[row][7] = C040
	row += 1

fp.close()

print(str(X) + "\n")
print(str(Y))