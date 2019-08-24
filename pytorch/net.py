class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # wx+b
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # x:[b,1,28,28]
        # h1 = relu(x*w1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(w2*h1+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2*w3+b3
        x = self.fc3(x)

        return x

for epoch in range(3):

    for batch_idx, (x.shape, y.shape) in enumerate(train_loader):
        print(x.shape, y.shape)
        # x:[b,1,28,28], y:[512]
        break