

# Insert the path of the file that contains the data
data = "Data/header_test.vcf"
info_individuals = "Data/integrated_call_samples_v3.20130502.ALL.panel"

# Create a dictionary for storing the continent of every individual
individuals_dict = {}

# Read the information database file for storing the procedence of every individual
with open(info_individuals) as fin:
    for line in fin:
        # Ignore reading the first line
        if line.startswith("sample"):
            continue
        else:
            # Store in individuals_dict the continent of every individual
            individuals_dict[line.split('\t')[0]] = line.split('\t')[2]


# 1 -- EAS (East Asia)
# 2 -- SAS (South Asia)
# 3 -- EUR (Europe)
# 4 -- AMR (America)
# 5 -- AFR (Africa)
mapping_dict = {}
mapping_dict['EAS'] = 1
mapping_dict['SAS'] = 2
mapping_dict['EUR'] = 3
mapping_dict['AMR'] = 4
mapping_dict['AFR'] = 5

with open(data) as fin, open("Data/individuals_labels.dat","a") as fout:
    for line in fin:
        if line.startswith("ID"):
            for element in line.split("\t"):
                if element == "ID":
                    continue
                else:
                    continent = individuals_dict[element.strip('\n')]
                    fout.write("%s\t"%(str(mapping_dict[continent])))
