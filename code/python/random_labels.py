import random



def scramble_labels(real_labels,identifier,index):
    new_labels = random.sample(real_labels,len(real_labels))
    f = open('labels/randomlabels_%s_%s.dat'%(str(identifier),str(index)),'w')
    for label in new_labels:
        f.write('%s\t'%(str(label)))
    f.write('\n')
    f.close()

datalabels_files = ['individuals_labels_TESTSET.dat', 'individuals_labels_TRAININGSET.dat']

for datafile in datalabels_files:
    identifier = datafile.split('_')[2]
    with open(datafile) as fin:
        line = fin.readlines()[0].strip('\n')
        real_labels = line.split('\t')
        for i in range(1000):
            scramble_labels(real_labels,identifier,i)
