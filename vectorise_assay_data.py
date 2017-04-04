import numpy
import h5py


def data_to_hdf5(input_filename, output_filename, y_genes, train_filename='_'):

    out_f = h5py.File(output_filename, 'a')

    #################################
    # EXTRACT GENE AND COMPOUND IDS #
    #################################

    # Identify all possible gene targets

    if train_filename == '_':

        genes = set()

        with open(input_filename) as f:
            for line in f:

                # Split line determine compound
                line_split = line.split()
                if line_split[1] not in genes:
                    genes.add(line_split[1])

        print('Total number of assays: ', len(genes))

        # Split X and Y genes

        for y_gene in y_genes:
            genes.remove(y_gene)

        x_genes = list(genes)

        print('Assay known number: ', len(y_genes))
        print('Assay predict number: ', len(x_genes))

    else:
        in_f = h5py.File(train_filename, 'a')

        x_genes_np = in_f['x_genes'][...]
        y_genes_np = in_f['y_genes'][...]

        x_genes = [str(x_gene, 'utf8') for x_gene in x_genes_np]
        y_genes = [str(y_gene, 'utf8') for y_gene in y_genes_np]

    # Identify all compounds

    compounds = set()

    with open(input_filename) as f:
        for line in f:

            # Split line determine compounds
            line_split = line.split()
            if line_split[0] not in compounds:
                compounds.add(line_split[0])

    compounds = list(compounds)

    print('compounds number', len(compounds))

    # Encode and write gene and compounds IDs to hdf5 file

    compounds_np = numpy.asarray([bytes(ins, encoding='utf8') for ins in compounds])
    x_genes_np = numpy.asarray([bytes(ins, encoding='utf8') for ins in x_genes])
    y_genes_np = numpy.asarray([bytes(ins, encoding='utf8') for ins in y_genes])

    out_f.create_dataset('compounds', data=compounds_np)
    out_f.create_dataset('x_genes', data=x_genes_np)
    out_f.create_dataset('y_genes', data=y_genes_np)

    ############################
    # WRITE BINARY DATA MATRIX #
    ############################

    # Create Dictionaries to index matrix row and column by compounds and gene name

    x_dict = {}
    for i in range(len(x_genes)):
        x_dict[x_genes[i]] = i

    y_dict = {}
    for i in range(len(y_genes)):
        y_dict[y_genes[i]] = i

    c_dict = {}
    for i in range(len(compounds)):
        c_dict[compounds[i]] = i

    x_activity = numpy.zeros((len(compounds), len(x_genes)), dtype='int8')  # X matrix
    y_activity = numpy.zeros((len(compounds), len(y_genes)), dtype='int8')  # Y matrix
    c = 0

    with open(input_filename) as f:
        for line in f:

            c += 1
            if c % 1000000 == 0:
                print(c)  # Track progress

            # Assign activity

            line_split = line.split()

            if line_split[1] in x_dict:  # Assign data to X_matrix

                col_pos = x_dict[line_split[1]]
                row_pos = c_dict[line_split[0]]

                # Option to include information on negative hit?
                # if line_split[2] == '0':
                #    x_activity[row_pos, col_pos] = -1

                if line_split[2] == '1':
                    x_activity[row_pos, col_pos] = 1

            if line_split[1] in y_dict:  # Assign data to Y_matrix

                col_pos = y_dict[line_split[1]]
                row_pos = c_dict[line_split[0]]

                # Option to include information on negative hit?
                # if line_split[2] == '0':
                #    y_activity[row_pos, col_pos] = -1

                if line_split[2] == '1':
                    y_activity[row_pos, col_pos] = 1

    # Save matrices to hdf5 file

    out_f.create_dataset('x_activity', data=x_activity)
    out_f.create_dataset('y_activity', data=y_activity)


y_gene_list = ['ATM', 'CHEK1', 'IL2', 'POLK', 'APOBEC3F', 'MAPK14', 'RGS4']
data_to_hdf5('activities_test.txt', 'activities_test.hdf5', y_gene_list, train_filename='activities_train.hdf5')
