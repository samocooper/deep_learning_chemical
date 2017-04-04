import h5py
import numpy

# Load set of compounds

in_f = h5py.File('activities_train.hdf5', 'r')
compounds_np = in_f['compounds'][...]
compounds = [str(ins, encoding='utf8') for ins in compounds_np]

c_dict = {}
for i in range(len(compounds)):
    c_dict[compounds[i]] = i

# Filter out features with less than 100 appearances

features = set()

with open('chemical_features.txt') as f:
    for line in f:
        # Split line
        line_split = line.split()
        if len(line_split) > 1:
            if line_split[1] not in features:
                features.add(line_split[1])

print('Total number of features: ', len(features))
features_list = list(features)

f_dict = {}
for i in range(len(features_list)):
    f_dict[features_list[i]] = 0

with open('chemical_features.txt') as f:
    for line in f:
        # Split line
        line_split = line.split()
        if len(line_split) > 1:
            f_dict[line_split[1]] += 1

for i in range(len(features_list)):
    if f_dict[features_list[i]] < 100:
        features.remove(features_list[i])

print('Filtered number of features: ', len(features))

# Determine maximum number of features for a chemical compound

prev = ''
count = 0
max_count = 0

with open('chemical_features.txt') as f:
    for line in f:
        # Split line determine compound
        line_split = line.split()
        if len(line_split) > 1:
            if line_split[1] in features:
                if not(line_split[0] == prev):
                    prev = line_split[0]
                    count = 0

                count += 1

                if count > max_count:
                    max_count = count
                    print(max_count)

print(max_count)

feature_matrix = numpy.zeros((len(compounds), max_count), dtype='uint16')
feature_counters = numpy.zeros((len(compounds)), dtype='uint16')

# Write feature index's to matrix

features_list = list(features)
f_dict = {}
for i in range(len(features_list)):
    f_dict[features_list[i]] = i

with open('chemical_features.txt') as f:
    for line in f:
        line_split = line.split()

        if len(line_split) > 1:
            if line_split[1] in features:
                c = c_dict[line_split[0]]
                feature_matrix[c, feature_counters[c]] = f_dict[line_split[1]]
                feature_counters[c] += 1

out_f = h5py.File('features_train.hdf5', 'a')

features_np = numpy.asarray([bytes(ins, encoding='utf8') for ins in features_list])
out_f.create_dataset('features', data=features_np)
out_f.create_dataset('x_features', data=feature_matrix)

