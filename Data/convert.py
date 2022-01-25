'''
Script to convert .txt to .csv files
'''

import argparse, os

parser = argparse.ArgumentParser(description='Convert txt to csv.')
parser.add_argument('-p', '--path', type=str, required=True , help="Folder containing all the txt files.")
args = parser.parse_args()

path = args.path

for file in os.listdir(path):
	file_path = os.path.join(path, file)

	if file.endswith('.txt'):

		csv_name = file_path.replace('\\', '/').rstrip('.txt') + '.csv'

		with open(file_path, 'r') as data:
			plaintext = data.read()

		plaintext = plaintext.replace(' ', ',')

		text_file = open(csv_name, "w")
		n = text_file.write(plaintext)
		text_file.close()
