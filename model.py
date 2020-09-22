import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Hyperparameters

BATCH_SIZE = 64
EPOCH_SIZE = 10
LEARNING_RATE = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data

transforms_fn = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms_fn, download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms_fn, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# network

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()

        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

model = Network().to(DEVICE)
optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.MSELoss()

# Train the model
for epoch in range(EPOCH_SIZE):
    for batch_Idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(DEVICE), target.to(DEVICE)
        data = data.view(-1, 28 * 28)
        en, de = model(data)
        loss = loss_function(de, data)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (batch_Idx+1) % 20 == 0:
            print("epoch:{},batch:{},loss:{:.4f}".format(epoch, batch_Idx+1, loss))

# save the model
torch.save(model, 'AutoEncoder.pkl')
print("save the model in AutoEncoder.pkl")

# load the model
model = torch.load('AutoEncoder.pkl')
print("load the model from AutoEncoder.pkl")

# 3D

view_data = train_dataset.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
encode, _ = model(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X = encode.data[:, 0].numpy()
Y = encode.data[:, 1].numpy()
Z = encode.data[:, 2].numpy()
labels = train_dataset.train_labels[:200].numpy()
for x, y, z, l in zip(X, Y, Z, labels):
    c = cm.rainbow(int(255 * l / 9))
    ax.text(x, y, z, l, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()


# compare the original and generated
plt.ion()
plt.show()

for i in range(10):
    test_data = train_dataset.data[i].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    _, result = model(test_data)

    # print('输入的数据的维度', train_data.train_data[i].size())
    # print('输出的结果的维度',result.size())

    im_result = result.view(28, 28)
    # print(im_result.size())
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.title('test_data')
    plt.imshow(train_dataset.data[i].numpy(), cmap='Greys')

    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.title('result_data')
    plt.imshow(im_result.detach().numpy(), cmap='Greys')
    plt.show()
    plt.pause(0.5)

plt.ioff()




