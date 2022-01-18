import matplotlib.pyplot as plt

f = open("RMSE.txt")

errors = f.read()

errors = errors.split("\n")
errors.pop()
errors = [float(error) for error in errors]

plt.xlabel("step")
plt.ylabel("RMS alingmnet error")
plt.plot(errors)

plt.show()
