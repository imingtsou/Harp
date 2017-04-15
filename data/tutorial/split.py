inFile = open("iris.data", "r")
out1 = open("iris.1", "w")
out2 = open("iris.2", "w")

flag = 0
for line in inFile:
	if flag == 0:
		out1.write(line)
	else:
		out2.write(line)
	flag = (flag + 1) % 2