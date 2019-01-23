#!/usr/bin/python

"""
Author: Elena Garcia
Course: Machine Learning
"""

import os


# Insert the path of the file that contains the data
data = "chr1.vcf"


def select_first_lines():
	# It will process and the file line-by-line, this way it will save memory
	with open(data) as fin, open("first_lines.vcf","a") as fout:
	    n = 0
	    for line in fin:
	        # Ignoring lines that start with ##
			# if line.startswith("##"):
	        #     continue
	        if n < 1000:
	            fout.write(line +"\n")
	            n += 1

	    fout.close()

def preprocessing():
	# # Insert the path of the file that contains the first lines of data
	# # Use only if select_first_lines() is deactivated
	# data = "../Initial data/first_lines.vcf"

	# It will process and the file line-by-line, this way it will save memory
	with open(data) as file, open("first_lines_preprocessed.vcf","a") as endfile:
		for line in file:
			# Ignoring lines that start with ##
				if line.startswith("1"):
					l = []
					# Keep only ID column
					ID = line.strip().split("\t")[2]
					# Keep only variables
					l = line.strip().split("\t")[9:]

					# Check the variables and change them
					for n1, x in enumerate(l):
						if x == "0|0":
							l[n1] = str(0)
						if x == "1|0" or x == "0|1":
							l[n1] = str(0.5)
						if x == "1|1":
							l[n1] = str(0)

					# Checking if on variable contains only 0 and removes it if it does
					if all(x.strip()==l[0].strip() for x in l) == False:
						l.insert(0, ID)
					endfile.write("\t".join(l)+"\n")


def main():
	select_first_lines()
	preprocessing()

	# end_time = datetime.now()
	# print('Analysis: {}'.format(end_time - start_time))
	# os.system('say "done"')


if __name__ == "__main__":
	main()
