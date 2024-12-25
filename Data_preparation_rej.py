

from sklearn.model_selection import train_test_split


def read_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []


    #with open('/content/gdrive/My Drive/Cation_anion_datasets/ident-60/SwissProt_ident_60_train.csv') as f:
    with open('./Dataset/inorganic_uniprot_ident100_t10_train.csv') as f:

      next(f)
      data_train = f.readlines()
      for d in data_train:
        d = d.split(',')
        x_train.append(d[0])
        y_train.append(int(d[1].strip('\n')))




    #with open('/content/gdrive/My Drive/Cation_anion_datasets/ident-60/SwissProt_ident_60_test.csv') as f:
    with open('./Dataset/inorganic_uniprot_ident100_t10_test.csv') as f:
      next(f)
      data_test = f.readlines()
      for d in data_test:
        d = d.split(',')
        x_test.append(d[0])
        y_test.append(int(d[1].strip('\n')))


    #with open('/content/gdrive/My Drive/Cation_anion_datasets/ident-60/SwissProt_ident_60_validation.csv') as f:
    with open('./Dataset/inorganic_uniprot_ident100_t10_validation.csv') as f:
      next(f)
      data_val = f.readlines()
      for d in data_val:
        d = d.split(',')
        x_val.append(d[0])
        y_val.append(int(d[1].strip('\n')))


    return x_train, x_test, x_val, y_train, y_test, y_val





X_train, X_test, X_val, y_train, y_test, y_val = read_dataset()



