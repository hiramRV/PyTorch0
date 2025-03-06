import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main(CPU=False):
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Force CPU for testing
    if(CPU): device=torch.device("cpu")
    ## True if the benchmark size is stable. 
    torch.backends.cudnn.benchmark = True

    # Read data, convert to NumPy arrays
    data = pd.read_csv("data/solar/sonar.all-data", header=None)
    X = data.iloc[:, 0:60].values
    y = data.iloc[:, 60].values

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # convert into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    # train-test split for evaluation of the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    #We move Test to GPU
    X_test, y_test= X_test.to(device), y_test.to(device)

    params = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 4}

    # set up DataLoader for training set
    loader = DataLoader(list(zip(X_train, y_train)), **params)

    # create model
    model = nn.Sequential(
        nn.Linear(60, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 1),
        nn.Sigmoid()
    )

    # Move model to GPU
    model = model.to(device)

    # Train the model
    n_epochs = 20
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # Load in GPU
            local_x, local_y= X_batch.to(device), y_batch.to(device)
            # Predict
            y_pred = model(local_x)
            loss = loss_fn(y_pred, local_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluate accuracy after training
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()
    acc = float(acc)
    print("Model accuracy: %.2f%%" % (acc*100))

# Run the code 
if __name__ == '__main__':
    #freeze_support()
    import time
    start_time = time.time()
    main(True)
    print("CPU: --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    main(False)
    print("GPU: --- %s seconds ---" % (time.time() - start_time))
    