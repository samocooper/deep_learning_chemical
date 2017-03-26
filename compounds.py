import sys
import numpy as np
import pickle
from scipy.sparse import csc_matrix
from scipy.cluster.vq import whiten, kmeans
import os.path
import threading


def find_dimensions(in_file):
    num_features = 0
    compounds = set()
    counter = 0

    with open(in_file) as infile:
        for line in infile:
            counter += 1
            if counter % 1000000 == 0:
                print(counter)
            feat = line.split()[1]
            tmp = feat.split('_')
            if len(tmp) > 1:
                n = int(tmp[1])
                if n > num_features:
                    num_features = n

            compound = line.split()[0]
            if compound != 'compound':
                if compound not in compounds:
                    compounds.add(compound)

    num_compounds = len(compounds)

    return num_compounds, num_features


def build_matrix():
    print('creating matrix...')
    counter = 0
    compounds = set()
    num_compounds = 0
    c_names = []
    col = []
    row = []
    data = []

    shape = find_dimensions(sys.argv[1])

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
    print(len(row))
    print(len(col))

    matrix = csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(shape[0], shape[1]))
    pickle.dump(matrix, open('csc_matrix-' + sys.argv[1] + '.pkl', 'wb'))
    return matrix


def cluster(matrix):
    whitened = whiten(matrix.todense())

    # for x in range(25, 40):
    #     means, distortion = kmeans(whitened, x)
    #     print distortion

    means, distortion = kmeans(whitened, 30)

    pickle.dump(means, open('30means-' + sys.argv[1] + '.pkl', 'wb'))

    return means, distortion


def drop_empty(matrix):
    print("Trying to remove empty columns...")

    indices = np.nonzero(matrix)
    columns_non_unique = indices[1]
    unique_columns = sorted(set(columns_non_unique))
    new_mat = matrix.tocsc()[:, unique_columns]
    pickle.dump(new_mat, open('no_zero_cols-' + sys.argv[1] + '.pkl'.format(sys.argv[1]), 'wb'))
    print('Old shape: ' + str(matrix.shape))
    print('New shape: ' + str(new_mat.shape))
    return new_mat


def main():
    if os.path.isfile('csc_matrix' + sys.argv[1] + '.pkl'):
        matrix = pickle.load('csc_matrix-' + sys.argv[1] + '.pkl', 'rb')
    else:
        matrix = build_matrix()

    new_mat = drop_empty(matrix)
    cluster(new_mat)

    # Threading:
    # t1 = threading.Thread(target=drop_empty(matrix))
    # t2 = threading.Thread(target=cluster(matrix))
    #
    # t1.start()
    # t2.start()
    #
    # t1.join()
    # t2.join()
    # , shape=(651058, 2532167)


if __name__ == '__main__':
    main()
