"""
net
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pickle as pk

from sklearn.metrics import roc_auc_score, precision_score,recall_score,f1_score, accuracy_score
def run(batchsize,lr,weightdecay):
    def get_precision(label, pred):
        pred = np.argmax(pred, axis=1)
        # print(pred.shape)
        label = np.argmax(label, axis=1)
        return precision_score(label, pred, average='macro')

    def get_recall(label, pred):
        pred = np.argmax(pred, axis=1)
        # print(pred.shape)
        label = np.argmax(label, axis=1)
        return recall_score(label, pred, average='macro')

    def get_f1_score(label, pred):
        pred = np.argmax(pred, axis=1)
        # print(pred.shape)
        label = np.argmax(label, axis=1)
        return f1_score(label, pred, average='macro')


    def get_acc(label, pred):
        pred = np.argmax(pred, axis=1)
        # print(pred.shape)
        label = np.argmax(label, axis=1)
        return accuracy_score(label, pred)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.line1 = nn.Linear(20, 64)
            self.line2 = nn.Linear(64, 128)
            self.line3 = nn.Linear(128, 9)

        def forward(self, x):
            x = self.line1(x)
            x = F.relu(x)
            x = F.dropout(x,0.2, training=self.training)

            x = self.line2(x)
            x = F.relu(x)


            x = self.line3(x)
            x = F.relu(x)
            x = F.dropout(x,0.2, training=self.training)
            return x


    class CustomDataset(Dataset):
        def __init__(self, data):
            # TODO
            feature = data['pdb_feature']
            lst = []
            for item in range(len(feature)):
                lst.append(feature[item])
            self.feature = np.array(lst)
            self.label = np.array(data['labels']).astype(np.int64)

        def __getitem__(self, idx):
            # TODO
            return self.feature[idx], self.label[idx]

        def __len__(self):
            # You should change 0 to the total size of your dataset.
            return len(self.feature)


    train_df = pk.load(open('/home/jiajunh/3CNN+protr/kfold_5/all_train.pkl', 'rb'))
    test_df = pk.load(open('/home/jiajunh/3CNN+protr/kfold_5/all_test.pkl', 'rb'))

    train_loader = DataLoader(CustomDataset(train_df), batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(CustomDataset(test_df), batch_size=batchsize)

    model = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightdecay)


    def train():
        loss_list = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion = torch.nn.CrossEntropyLoss()
            output = model(data.float())
            loss = criterion(output, target)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        return sum(loss_list) / len(loss_list)


    def test():
        loss_list = []
        model.eval()
        label = []
        preds = []

        for _, (data, target) in enumerate(test_loader):
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                # optimizer.zero_grad()
                label.append(F.one_hot(target, num_classes=9).cpu().numpy())
                criterion = torch.nn.CrossEntropyLoss()
                output = model(data.float())
                pred = F.softmax(output, dim=1)
                preds.append(pred.cpu().numpy())
                loss = criterion(output, target)
                loss_list.append(loss.item())
        preds = np.vstack(preds)
        label = np.vstack(label)

        return sum(loss_list) / len(loss_list), roc_auc_score(label, preds, average='micro'), get_precision(label,preds),get_recall(label,preds),get_f1_score(label,preds), get_acc(label, preds)


    for epoch in range(1, 300):
        train_loss = train()
        test_loss, auc_score, precision,recall,f1, acc = test()
    print('batchsize:{:03d},lr: {:.5f},weightdecay: {:.6f},Epoch: {:03d}, Train Loss: {:.4f}, Test Loss: {:.4f}, AUC: {:.3f}, Precision:{:3f},Recall:{:3f},F1:{:3f}, ACC:{:3f}'.format(batchsize,lr,weightdecay,epoch, train_loss, test_loss, auc_score,precision,recall, f1, acc))

run(32,0.01,0.001)
run(64,0.01,0.001)
run(128,0.01,0.001)

run(32,0.001,0.001)
run(64,0.001,0.001)
run(128,0.001,0.001)

run(32,0.0001,0.001)
run(64,0.0001,0.001)
run(128,0.0001,0.001)

run(32,0.01,0.0001)
run(64,0.01,0.0001)
run(128,0.01,0.0001)

run(32,0.001,0.0001)
run(64,0.001,0.0001)
run(128,0.001,0.0001)

run(32,0.0001,0.0001)
run(64,0.0001,0.0001)
run(128,0.0001,0.0001)

run(32,0.01,0.00001)
run(64,0.01,0.00001)
run(128,0.01,0.00001)

run(32,0.001,0.00001)
run(64,0.001,0.00001)
run(128,0.001,0.00001)

run(32,0.0001,0.00001)
run(64,0.0001,0.00001)
run(128,0.0001,0.00001)
