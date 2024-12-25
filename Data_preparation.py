
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



def read_miscellaneous():
    misc_x = []
    misc_y = []

    with open('./Dataset/Uniprot_ICAT_100_t10_mis_downsampled_tuple.txt', 'r') as f:
      dataset = f.readlines()
      for item in dataset:
        item = item.strip('(').strip(')').strip('\n').strip('\'')
        if int(item.split(',')[1].strip(')')) == 0:
          x = ' '.join(item.split(',')[0])
          x = x.replace('U', 'X')
          x = x.replace('Z', 'X')
          x = x.replace('O', 'X')
          x = x.replace('B', 'X')

          misc_x.append(x)
          misc_y.append(12)
    return misc_x, misc_y


x_train, x_test, x_val, y_train, y_test, y_val = read_dataset()
misc_x, misc_y = read_miscellaneous()



#split miscellaneous
X_train_mis, X_test_mis, y_train_mis, y_test_mis = train_test_split(misc_x, misc_y,
                                                    stratify=misc_y,
                                                    test_size=0.2)
X_train_mis, X_val_mis, y_train_mis, y_val_mis = train_test_split(X_train_mis, y_train_mis,
                                                    stratify=y_train_mis,
                                                    test_size=0.2)

       
#concate datasets

X_train = x_train + X_train_mis
X_test = x_test + X_test_mis
X_val = x_val + X_val_mis

y_train = y_train + y_train_mis
y_test = y_test + y_test_mis
y_val = y_val + y_val_mis
