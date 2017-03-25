
def read_file(f, func):
    with open(f) as infile:
        for line in infile:
            func(line)


class CompoundNormalization(object):
    def __init__(self, f):
        self.max = 0
        self.compounds = []
        self.run(f)

    def run(self, f):
        print "Finding number of compound features..."
        self.read_file(f, self.find_max)

    def find_max(self, line):
        feat = line.split()[1]
        tmp = feat.split('_')
        if len(tmp) > 1:
            n = int(tmp[1])
            if n > self.max:
                self.max = n

    def create_compound_list(self, line):
        compound = line.split()[0]
        if compound != 'compound':
            if compound not in self.compounds:
                self.compounds.append(compound)





if __name__ == '__main__':
    c = CompoundNormalization("chemical_features.txt")
    print c.max
    print len(c.compounds)

# Find common features
