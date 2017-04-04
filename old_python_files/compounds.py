import sys
import numpy as np
import h5py
import pickle
from scipy.sparse import csr_matrix
from scipy.cluster.vq import whiten, kmeans
import os.path



def cluster(matrix):
    whitened = whiten(matrix.todense())

    # for x in range(25, 40):
    #     means, distortion = kmeans(whitened, x)
    #     print distortion

    means, distortion = kmeans(whitened, 30)

    # pickle.dump(means, open('30means-' + sys.argv[1] + '.pkl', 'wb'))

    return means, distortion


def drop_empty(matrix):
    print("Trying to remove empty columns...")

    indices = np.nonzero(matrix)
    columns_non_unique = indices[1]
    unique_columns = set(columns_non_unique)
    new_mat = matrix.tocsc()[:, unique_columns]
    # pickle.dump(new_mat, open('no_zero_cols-' + sys.argv[1] + '.pkl'.format(sys.argv[1]), 'wb'))
    print('Old shape: ' + str(matrix.shape))
    print('New shape: ' + str(new_mat.shape))
    return new_mat


def pca(matrix):
    clf = TruncatedSVD(1000)
    matrix_pca = clf.fit_transform(matrix)
    pickle.dump(matrix_pca, open('PCA-' + sys.argv[1] + '.pkl'.format(sys.argv[1]), 'wb'))
    print(str(matrix_pca.shape))
    print(matrix_pca)
    return matrix_pca


def main():
    # if os.path.isfile('feature_matrix.h5'):
    #     f = h5py.File('feature_matrix.h5', 'r')
    #     matrix = f['matrix'][:]
    # else:
    f = h5py.File('feature_matrix.h5', 'r')
    data = f['data'][:]
    col = f['col'][:]
    row = f['row'][:]
    shape = f['shape'][:]

    matrix = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(shape[0], shape[1]),
                        dtype=np.uint8)
    sums = matrix.sum(axis=0)
    print(sums)
    print(min(sums[0, :]))

    filtered_mat = matrix[:, np.where(sums > 10000)[1]]
    print('new: ' + str(filtered_mat.shape))
    print('old: ' + str(matrix.shape))

    fo = h5py.File('filtered_matrix.h5', 'w')

    fo.create_dataset('data', data=filtered_mat.data)
    fo.create_dataset('indices', data=filtered_mat.indices)
    fo.create_dataset('indptr', data=filtered_mat.indptr)





    # new_mat = drop_empty(matrix)
    # cluster(new_mat)
    # pca(new_mat)

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
