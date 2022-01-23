import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Create plot from logged errors')
parser.add_argument('-p', '--path', type=str, required=True , help="File containing the logs")
args = parser.parse_args()

path = args.path

f = open(path)

errors = f.read()

errors = errors.split("\n")
errors.pop()
errors = [float(error) for error in errors]

plt.xlabel("step")
plt.ylabel("Error")
plt.plot(errors)

plt.show()
