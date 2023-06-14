import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# using dataframe here just to make it clearer which columns go into each preprocessing stage
train_data = pd.read_csv("adult.data", sep=', ', header=0, engine='python')
test_data = pd.read_csv("adult.test", sep=', ', header=0, engine='python')

data_dims = 0 # will hold final number of dimensions after encoding

def preprocess_data(table):
    # grab ordinal categories, one-hot categories, and numerical categories
    X_cats = table.get(['workclass', 'education'])
    X_hots = table.get(['marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
    X_nums = table.get(['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week'])
    y_raw = table.get('income')

    # category orders for the ordinal encoder
    # arguable how much is gained from making these ordinal - I believe there should
    # be a meaningful gradient, but if I had any discipline I would explore this
    # with an EDA :)
    emp_cats = ['?','Never-worked','Without-pay','Local-gov','State-gov','Federal-gov','Self-emp-not-inc','Self-emp-inc','Private']
    edu_cats = ['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th','12th','HS-grad','Some-college','Assoc-acdm','Assoc-voc','Prof-school','Bachelors','Masters','Doctorate']
    # test_data is missing "Holand-Netherlands" as a value in native-country
    # only appears in a single row in train_data, so I just dropped that row
    # also, categories appear in different orders in each data set
    # rather than sort + hallucinate filler row(s), I just hard code categories for one-hot
    country_cats = ['?' 'Cambodia' 'Canada' 'China' 'Columbia' 'Cuba' 'Dominican-Republic'
 'Ecuador' 'El-Salvador' 'England' 'France' 'Germany' 'Greece' 'Guatemala'
 'Haiti' 'Honduras' 'Hong' 'Hungary' 'India' 'Iran'
 'Ireland' 'Italy' 'Jamaica' 'Japan' 'Laos' 'Mexico' 'Nicaragua'
 'Outlying-US(Guam-USVI-etc)' 'Peru' 'Philippines' 'Poland' 'Portugal'
 'Puerto-Rico' 'Scotland' 'South' 'Taiwan' 'Thailand' 'Trinadad&Tobago'
 'United-States' 'Vietnam' 'Yugoslavia']

    X_cats_enc = OrdinalEncoder(categories=[emp_cats, edu_cats]).fit_transform(X_cats)
    X_hots_enc = OneHotEncoder(sparse_output=False).fit_transform(X_hots)
    X_nums_scaled = RobustScaler().fit_transform(X_nums)
    y = LabelEncoder().fit_transform(y_raw)

    # combine all the preprocessed values back into one big table
    X = np.hstack((X_nums_scaled, X_cats_enc, X_hots_enc))
    X = np.float32(X)
    y = np.expand_dims(y, axis=1)
    print(X.shape, y.shape)  # (32561, 85)
    print(type(X[0][0]), type(y[0]))

    # hacky way to pass dimension size around our script
    global data_dims
    data_dims = len(X[0])

    # make processed data available to torch
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    return data_utils.DataLoader(data_utils.TensorDataset(X_tensor, y_tensor), batch_size=64)

train_dataloader = preprocess_data(train_data)
test_dataloader = preprocess_data(test_data)

# torch stuff now - defining a generic stack
class NeuralNet(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.CELU(),
            nn.Linear(512, 512),
            nn.CELU(),
            nn.Linear(512, 512),
            nn.CELU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

# since we generate a lot of one-hot columns, size the model based on the processed data shape
model = NeuralNet(dims=data_dims).to(device)
print(model)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0.925)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

# most of this shamelessly ripped from the pytorch docs, tweaked later
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_func):
#    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    metric_acc = BinaryAccuracy().to(device)
    metric_pre = BinaryPrecision().to(device)
    model.eval()
    # test metrics calculate on a per-batch basis, so we average across batches
    test_loss, accuracy, precision = 0,0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            accuracy += metric_acc(preds=pred, target=y)
            precision += metric_pre(preds=pred, target=y)
    test_loss /= num_batches
    accuracy /= num_batches
    precision /= num_batches
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Precision: {(100*precision):>0.1f}%, Avg loss: {test_loss:>8f} \n")

num_epochs = 500
for epoch in range(num_epochs):
    print(f"epoch number {epoch}")
    train(train_dataloader, model, loss_func, optimizer)
    test(test_dataloader, model, loss_func)
    scheduler.step()
print("done.")