import math

#data = "Data/header_test.vcf"
data = "Data/ch1_SNP_Data.vcf"

labels = "Data/individuals_labels.dat"

# Compute the entropy from a list of counts
def calc_entropy(count):
    #print count
    entropy = 0
    test = 0
    for possibility in count:
        probability = float(possibility)/float(sum(count))
        test += probability
        # avoid doing the log of 0 if there is not seen any specific event
        if probability == 0:
            continue
        else:
            entropy -= probability*math.log(probability,2)
    return entropy

def IG(snp_list,labels):
    #print snp_list
    snp = snp_list[0]
    snp_list.pop(0)
    # List for storing the counts (possible values: 0, 0.5 and 1)
    count = [0,0,0]
    for element in snp_list:
        if element == '0':
            count[0] += 1
        elif element == '0.5':
            count[1] += 1
        elif element == '1':
            count[2] += 1
    entropy = calc_entropy(count)


# 1 -- EAS (East Asia)
# 2 -- SAS (South Asia)
# 3 -- EUR (Europe)
# 4 -- AMR (America)
# 5 -- AFR (Africa)
    probability_continent = []
    for continent in [1,2,3,4,5]:
        probability_continent.append(float(labels.count(continent)/len(labels)))
    indexes_1 = [i for i,x in enumerate(labels) if x == '1']
    indexes_2 = [i for i,x in enumerate(labels) if x == '2']
    indexes_3 = [i for i,x in enumerate(labels) if x == '3']
    indexes_4 = [i for i,x in enumerate(labels) if x == '4']
    indexes_5 = [i for i,x in enumerate(labels) if x == '5']

    specific_conditional_entropy = []
    for continent in [1,2,3,4,5]:
        continent_snp = []
        continent_snp_count = []
        for position in indexes_1:
            continent_snp.append(snp_list[position])
        #print indexes_1
        #print continent_snp
        #print len(continent_snp)
        continent_snp_count.append(float(continent_snp.count(('0'))))
        continent_snp_count.append(float(continent_snp.count(('0.5'))))
        continent_snp_count.append(float(continent_snp.count(('1'))))
        specific_conditional_entropy.append(calc_entropy(continent_snp_count))

    conditional_entropy = 0
    for pos in range(len(specific_conditional_entropy)):
        conditional_entropy += probability_continent[pos]*specific_conditional_entropy[pos]

    IG = entropy - conditional_entropy
    return snp, IG

with open(labels) as fin:
    labels_list = []
    for line in fin:
        labels_list = line.split('\t')

with open(data) as fin:
    for line in fin:
        # Ignore reading the first line
        if line.startswith("ID"):
            continue
        else:
            snp, ig = IG(line.split('\t'),labels_list)
            print snp + '   ' + str(ig)
