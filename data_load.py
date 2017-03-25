import _pickle
import numpy

target = set()

# Identify all possible gene targets write to file
'''
with open('activities_train.txt') as f:
    for line in f:

        # Split line determine compound
        line_split = line.split()
        if line_split[1] not in target:
            target.add(line_split[1])


print(len(target))
_pickle.dump(target, open('all_target.pkl', 'wb'))
'''
# Identify all compounds and write to file
'''
with open('activities_train.txt') as f:
    for line in f:

        # Split line determine compound
        line_split = line.split()
        if line_split[0] not in target:
            target.add(line_split[0])


print(len(target))
_pickle.dump(target, open('all_compound.pkl', 'wb'))

'''
# Split X and Y genes

'''

target = _pickle.load(open('all_target.pkl', 'rb'))
compound = _pickle.load(open('all_compound.pkl', 'rb'))

y_genes = ['ATM','CHEK1','IL2','POLK','APOBEC3F','MAPK14','RGS4']

for gene in y_genes:
    target.remove(gene)

x_genes = list(target)

_pickle.dump(x_genes, open('x_target.pkl', 'wb'))
_pickle.dump(y_genes, open('y_target.pkl', 'wb'))
'''


# Split data into x and y activity records
'''

x_genes = _pickle.load(open('x_target.pkl', 'rb'))
y_genes = _pickle.load(open('y_target.pkl', 'rb'))

compound = _pickle.load(open('all_compound.pkl', 'rb'))
compound = list(compound)

x_dict = {}
for i in range(len(x_genes)):
    x_dict[x_genes[i]] = i

y_dict = {}
for i in range(len(y_genes)):
    y_dict[y_genes[i]] = i

c_dict = {}
for i in range(len(compound)):
    c_dict[compound[i]] = i

x_activity = numpy.zeros((len(compound), len(x_genes)), dtype='int8')
y_activity = numpy.zeros((len(compound), len(y_genes)), dtype='int8')

c = 0

with open('activities_train.txt') as f:
    for line in f:
        c += 1
        if c % 100000 == 0:
            print(c)
        # Assign activity

        line_split = line.split()

        if line_split[1] in x_dict:

            col_pos = x_dict[line_split[1]]
            row_pos = c_dict[line_split[0]]

            if line_split[2] == '0':
                x_activity[row_pos, col_pos] = -1

            if line_split[2] == '1':
                x_activity[row_pos, col_pos] = 1

        if line_split[1] in y_dict:

            col_pos = y_dict[line_split[1]]
            row_pos = c_dict[line_split[0]]

            if line_split[2] == '0':
                y_activity[row_pos, col_pos] = -1

            if line_split[2] == '1':
                y_activity[row_pos, col_pos] = 1


print(sum(y_activity.flatten()))

_pickle.dump(x_activity, open('x_activity.pkl', 'wb'))
_pickle.dump(y_activity, open('y_activity.pkl', 'wb'))

'''

x_activity = _pickle.load(open('x_activity.pkl', 'rb'))
y_activity = _pickle.load(open('y_activity.pkl', 'rb'))
