import sys
import numpy as np
import h5py
import pickle

def build_matrix():
    print('creating matrix...')
    counter = 0
    compounds = set()
    num_compounds = 0
    c_names = []
    col = []
    row = []
    data = []

    # shape = find_dimensions(sys.argv[1])

    with open(sys.argv[1]) as infile:
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
            col.append(feat - 1)
            row.append(num_compounds - 1)

    print(len(data))
    print(row[-1] + 1)
    print(max(col) + 1)

    return np.array(data), np.array(row), np.array(col), c_names
    # pickle.dump(matrix, open('csc_matrix-' + sys.argv[1] + '.pkl', 'wb'))

def main():
    # if os.path.isfile('feature_matrix.h5'):
    #     f = h5py.File('feature_matrix.h5', 'r')
    #     matrix = f['matrix'][:]
    # else:
    data, row, col, names = build_matrix()
    shape = [row[-1] + 1, max(col) + 1]
    f = h5py.File('feature_matrix.h5', 'w')
    f.create_dataset('data', data=data)
    f.create_dataset('col', data=col)
    f.create_dataset('row', data=row)
    f.create_dataset('shape', data=shape)
    pickle.dump(names, open('names.txt', 'wb'))

if __name__ == '__main__':
    main()
