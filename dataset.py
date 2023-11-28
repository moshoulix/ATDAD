"""
label    Benign   Anomaly
-------------------------
Train    523666      0
Test1a             5000
Test1b    5000
Test2     5000     5000
Test3     5000     5000

"""
from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np


class IDSDataset(Dataset):
    def __init__(self, npz_file, name, train_num=None):
        self.data_df = np.load(npz_file)[name]
        self.name = name
        self.train_num = len(self.data_df) if train_num is None else train_num

    def __len__(self):
        return self.train_num

    def __getitem__(self, index):
        temp = self.data_df[index]
        flow_values = torch.FloatTensor(temp[:-1]) * 2 - 1
        label = temp[-1]
        tensor_label = torch.FloatTensor([0, 1]) if label else torch.FloatTensor([1, 0])
        return flow_values, label, tensor_label


def ids_dataloader(npz, name, batch_size, train_num=None, shuffle=True, drop_last=False):
    return DataLoader(IDSDataset(npz_file=npz, name=name, train_num=train_num),
                      batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


if __name__ == '__main__':
    import pandas as pd
    np.random.seed(1234)

    def timestamp2float(stamp):
        res = [0, 0, 0]
        res[0] = float(stamp[11]) * 10 + float(stamp[12])
        res[1] = float(stamp[14]) * 10 + float(stamp[15])
        res[2] = float(stamp[17]) * 10 + float(stamp[18])
        return res[0] * 3600 + res[1] * 60 + res[2]


    # load csv file
    df = pd.read_csv('./dataset/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv', low_memory=False)

    # delete text data
    for index, data in df['Flow Byts/s'].to_frame().itertuples():
        try:
            temp = float(data)
        except ValueError:
            # print(data)
            df.drop(labels=index, inplace=True)

    # benign: 0     the other: 1
    df.loc[df['Label'] != 'Benign', 'Label'] = 1
    df.loc[df['Label'] == 'Benign', 'Label'] = 0

    # one hot
    one_hot_protocol = pd.get_dummies(df['Protocol'])
    one_hot_protocol.columns = one_hot_protocol.columns.map(lambda x: 'Protocol_' + x)
    df.drop('Protocol', axis=1, inplace=True)
    df = pd.concat([one_hot_protocol, df], axis=1)

    # timestamp
    df['Timestamp'] = df['Timestamp'].apply(lambda x: timestamp2float(x))

    # delete nan & inf
    df = df.astype(float)
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)

    # delete columns that are all zeros
    df.drop(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
             'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
             'Bwd PSH Flags', 'Bwd URG Flags'], inplace=True, axis=1)

    # split dataset
    df_Benign = df.loc[df['Label'] == 0].copy()
    df_Benign.drop('Label', axis=1, inplace=True)
    df_Benign = df_Benign.reset_index(drop=True)

    df_Anomaly = df.loc[df['Label'] == 1].copy()
    df_Anomaly.drop('Label', axis=1, inplace=True)
    df_Anomaly = df_Anomaly.reset_index(drop=True)

    # normalization
    df_Benign = (df_Benign - np.mean(df_Benign)) / (np.std(df_Benign) + 1e-4)
    df_Anomaly = (df_Anomaly - np.mean(df_Benign)) / (np.std(df_Benign) + 1e-4)

    min_cols = df_Benign.min()
    max_cols = df_Benign.max()
    df_Benign = (df_Benign - min_cols) / (max_cols - min_cols)
    df_Anomaly = (df_Anomaly - min_cols) / (max_cols - min_cols)

    # To avoid incorporating information from the test set,
    # normalization only took into account the maximum and minimum values of the training set.
    # Test set data is directly truncated for out-of-bounds values to prevent overflow.
    df_Anomaly[df_Anomaly > 1] = 1
    df_Anomaly[df_Anomaly < 0] = 0

    # delete nan & inf
    df_Benign.replace(np.inf, np.nan, inplace=True)
    df_Anomaly.replace(np.inf, np.nan, inplace=True)

    # shuffle
    df_Benign = df_Benign.sample(frac=1)
    df_Anomaly = df_Anomaly.sample(frac=1)

    # add label column
    B_label = pd.Series([0 for i in range(len(df_Benign))])
    df_Benign = df_Benign.assign(Label=B_label)

    A_label = pd.Series([1 for i in range(len(df_Anomaly))])
    df_Anomaly = df_Anomaly.assign(Label=A_label)

    # check
    assert max(df_Anomaly.max().to_list()) <= 1 and min(df_Anomaly.min().to_list()) >= 0 and \
           max(df_Benign.max().to_list()) <= 1 and min(df_Benign.min().to_list()) >= 0 and \
           not df_Anomaly.isna().values.any() and not df_Benign.isna().values.any()

    Benign_ = np.array(df_Benign)
    Anomaly_ = np.array(df_Anomaly)
    np.savez_compressed('./dataset/handled_dataset.npz',
                        Benign=Benign_, Anomaly=Anomaly_)

    proportions = df['Label'].value_counts()
    print(proportions)
    print("Anomaly Percentage", proportions[1] / proportions.sum())  # Anomaly Percentage 0.11238550066546622

    print('number of Anomaly:', len(Anomaly_))  # 68236
    print('number of Benign:', len(Benign_))    # 538666

    # some datasets for tests
    Test2_ = pd.concat([pd.DataFrame(Anomaly_[5000:10000]), pd.DataFrame(Benign_[5000:10000])], axis=0)
    Test2_ = np.array(Test2_)
    Test3_ = pd.concat([pd.DataFrame(Anomaly_[10000:15000]), pd.DataFrame(Benign_[10000:15000])], axis=0)
    Test3_ = np.array(Test3_)
    np.savez_compressed('./dataset/handled_dataset2.npz',
                        Train=Benign_[15000:],
                        Test1a=Anomaly_[:5000], Test1b=Benign_[:5000],
                        Test2=Test2_,
                        Test3=Test3_)

    '''
    cols_to_norm = ['Dst Port', 'Timestamp',
                    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
                    'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
                    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
                    'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
                    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                    'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
                    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
                    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len',
                    'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
                    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
                    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
                    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
                    'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
                    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
                    'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
                    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
                    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
                    'Idle Min']
    '''
