import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

fp = open('breast-cancer.data', 'r')
X = np.zeros((277, 9))					#input Data
Y = np.zeros((277, 1))					#labels
row = 0

for line in fp.readlines():
	data = line.strip()
	data = line.split(',')

	classification = data[0]

	if(classification == "no-recurrence-events"):
		classification = 0
	elif(classification == "recurrence-events"):
		classification = 1

	age = data[1]

	if(age == "10-19"):
		age = 0
	elif(age == "20-29"):
		age = 1
	elif(age == "30-39"):
		age = 2
	elif(age == "40-49"):
		age = 3
	elif(age == "50-59"):
		age = 4
	elif(age == "60-69"):
		age = 5
	elif(age == "70-79"):
		age = 6
	elif(age == "80-89"):
		age = 7
	elif(age == "90-99"):
		age = 8

	menopause = data[2]

	if(menopause == "lt40"):
		menopause = 0
	elif(menopause == "ge40"):
		menopause = 1
	elif(menopause == "premeno"):
		menopause = 2

	tumor_size = data[3]

	if(tumor_size == "0-4"):
		tumor_size = 0
	elif(tumor_size == "5-9"):
		tumor_size = 1
	elif(tumor_size == "10-14"):
		tumor_size = 2
	elif(tumor_size == "15-19"):
		tumor_size = 3
	elif(tumor_size == "20-24"):
		tumor_size = 4
	elif(tumor_size == "25-29"):
		tumor_size = 5
	elif(tumor_size == "30-34"):
		tumor_size = 6
	elif(tumor_size == "35-39"):
		tumor_size = 7
	elif(tumor_size == "40-44"):
		tumor_size = 8
	elif(tumor_size == "45-49"):
		tumor_size = 9
	elif(tumor_size == "50-54"):
		tumor_size = 10
	elif(tumor_size == "55-59"):
		tumor_size = 11

	inv_nodes = data[4]

	if(inv_nodes == "0-2"):
		inv_nodes = 0
	elif(inv_nodes == "3-5"):
		inv_nodes = 1
	elif(inv_nodes == "6-8"):
		inv_nodes = 2
	elif(inv_nodes == "9-11"):
		inv_nodes = 3
	elif(inv_nodes == "12-14"):
		inv_nodes = 4
	elif(inv_nodes == "15-17"):
		inv_nodes = 5
	elif(inv_nodes == "18-20"):
		inv_nodes = 6
	elif(inv_nodes == "21-23"):
		inv_nodes = 7
	elif(inv_nodes == "24-26"):
		inv_nodes = 8
	elif(inv_nodes == "27-29"):
		inv_nodes = 9
	elif(inv_nodes == "30-32"):
		inv_nodes = 10
	elif(inv_nodes == "33-35"):
		inv_nodes = 11
	elif(inv_nodes == "36-39"):
		inv_nodes = 12

	node_caps = data[5]

	if(node_caps == "yes"):
		node_caps = 1
	elif(node_caps == "no"):
		node_caps = 0

	deg_malig = data[6]
	deg_malig = int(deg_malig)

	breast = data[7]

	if(breast == "left"):
		breast = 0
	elif(breast == "right"):
		breast = 1

	breast_quad = data[8]

	if(breast_quad == "left_up"):
		breast_quad = 0
	elif(breast_quad == "left_low"):
		breast_quad = 1
	elif(breast_quad == "right_up"):
		breast_quad = 2
	elif(breast_quad == "right_low"):
		breast_quad = 3
	elif(breast_quad == "central"):
		breast_quad = 4

	irradiat = data[9]
	if(irradiat == "yes\n" or irradiat == "yes"):
		irradiat = 1
	elif(irradiat == "no\n" or irradiat == "no"):
		irradiat = 0

	Y[row][0] = classification 
	X[row][0] = age
	X[row][1] = menopause
	X[row][2] = tumor_size
	X[row][3] = inv_nodes
	X[row][4] = node_caps
	X[row][5] = deg_malig
	X[row][6] = breast
	X[row][7] = breast_quad
	X[row][8] = irradiat

	row += 1

fp.close()

print(str(X) + "\n")
print(str(Y))