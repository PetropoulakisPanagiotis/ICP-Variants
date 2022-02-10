import matplotlib.pyplot as plt

f = open("./colors/52/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error1 = [float(error) for error in error]

f = open("./colors/11/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error2 = [float(error) for error in error]

f = open("./colors/53/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error3 = [float(error) for error in error]

f = open("./colors/55/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error4 = [float(error) for error in error]

f = open("./colors/29/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error5 = [float(error) for error in error]

f = open("./colors/30/RMSE0.txt")

error = f.read()

error = error.split("\n")
error.pop()
error6 = [float(error) for error in error]

plt.xlabel("Iteration")
plt.ylabel("RMS alignment error")
plt.plot(error1, label='Symmetric const weig. + Projective', marker="h")
plt.plot(error2, label='Symmetric colors weig. + Projective', marker=".")
plt.plot(error3, label='Symmetric constant weig. + Projective + Multi-Resolution', marker='x')
plt.plot(error4, label='Symmetric colors weig. + Projective + Multi-Resolution', marker="_")
#plt.plot(error2, label='Plane non-linear', marker="+")
#plt.plot(error3, label='Symmetric non-linear', marker="_")

plt.legend()
plt.show()
