import numpy as np
from scipy.sparse import csc_matrix


def read_file(f, func):
    with open(f) as infile:
        for line in infile:
            func(line)


class CompoundNormalization(object):
    def __init__(self, f):
        self.max = 0
        self.compounds = set()
        self.counter = 0
        self.run(f)

    def run(self, f):
        print("Finding number of compound features...")

        read_file(f, self.create_compound_list)

    def create_compound_list(self, line):
        self.counter += 1
        if self.counter % 1000000 == 0:
            print(self.counter)
        feat = line.split()[1]
        tmp = feat.split('_')
        if len(tmp) > 1:
            n = int(tmp[1])
            if n > self.max:
                self.max = n

        compound = line.split()[0]
        if compound != 'compound':
            if compound not in self.compounds:
                self.compounds.add(compound)


# Go through file
# For every compound add row
# for every feat add col and val (1)
if __name__ == '__main__':
    # normalized = CompoundNormalization("chemical_features.txt")
    # print(normalized.max)
    # print(len(normalized.compounds))
    # max = normalized.max
    # num_compounds = len(normalized.compounds)
    #

    print('creating matrix...')
    counter = 0
    compounds = set()
    num_compounds = 0
    c_names = []
    col = []
    row = []
    c = 0
    data = []
    with open("chemical_features.txt") as infile:
        for line in infile:
            if line.split()[0] == 'compound':
                continue

            counter += 1
            if counter % 1000000 == 0:
                print(counter)
            tmp = line.split()
            compound = tmp[0]
            feat = int(tmp[1].split('_')[1])

            if compound not in compounds:
                compounds.add(compound)
                c_names.append(compound)
                num_compounds += 1

            data.append(1)
            col.append(feat)
            row.append(num_compounds)

    print(len(data))
    print(len(row))
    print(len(col))
    csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(2532167, 651058))

# Find common features
