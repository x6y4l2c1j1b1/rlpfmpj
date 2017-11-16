import glob
import csv
import numpy as np
from collections import defaultdict

high_beta_path = r'/Users/xx/Desktop/Reinforce Learning/project/rlpfmpj/Data/rawdata/high_beta'
low_beta_path = r'/Users/xx/Desktop/Reinforce Learning/project/rlpfmpj/Data/rawdata/low_beta'
high_beta_file_names = glob.glob(high_beta_path+"/*.csv")
low_beta_file_names = glob.glob(low_beta_path+"/*.csv")

def get_common_date():
	date_dict = defaultdict(int)
	for filename in high_beta_file_names:
		with open(filename) as csvfile:
			readCSV = csv.reader(csvfile,delimiter=',')
			rows = [row for row in readCSV]
		for row in rows[1:]:
			date_dict[row[0]] += 1
	for filename in low_beta_file_names:
		with open(filename) as csvfile:
			readCSV = csv.reader(csvfile,delimiter=',')
			rows = [row for row in readCSV]
		for row in rows[1:]:
			date_dict[row[0]] += 1
	s = set()
	for date, cnt in date_dict.items():
		if cnt==20:
			s.add(date)
	return s

# s = get_common_date()
# print len(s)

def readData(date_set):
	"""
	only use close price as the value of the stock in each day

	return type:
		numpy array of 20 stocks's close price with respect to their common valid date
	"""
	price_array = np.zeros((20,len(date_set)))
	index = 0
	for filename in high_beta_file_names:
		with open(filename) as csvfile:
			print filename
			readCSV = csv.reader(csvfile,delimiter=',')
			rows = [row for row in readCSV][1:]
			date = [row[0] for row in rows]
			close_price = [row[4] for row in rows if row[0] in date_set]
			### valid from 2008-04-23 to 2017-10-13
			price_array[index] = np.array(close_price)
			index += 1
	for filename in low_beta_file_names:
		with open(filename) as csvfile:
			print filename
			readCSV = csv.reader(csvfile,delimiter=',')
			rows = [row for row in readCSV][1:]
			date = [row[0] for row in rows]
			close_price = [row[4] for row in rows if row[0] in date_set]
			### valid from 2008-04-23 to 2017-10-13
			price_array[index] = np.array(close_price)
			index += 1
	return np.array(price_array)

date_set = get_common_date()
print date_set
price_array = readData(date_set)
print(price_array)

