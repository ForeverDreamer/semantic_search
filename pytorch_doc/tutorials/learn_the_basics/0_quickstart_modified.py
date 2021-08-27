import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# 参考资料
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# https://research.zalando.com/project/fashion_mnist/fashion_mnist/
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

# 优化步骤
# 1.引入卷积神经网络和池化层
# 2.估计是两层池化之后信息量丢失太大，导致准确率太低，试试移除池化层（准确率飙升近30%！）
# 3.增加训练epoch（准确率提升7%）

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# # Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 训练阶段
batch_size = 64
# 测试阶段
# batch_size = 1

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print("Shape of X [N, C, H, W]: ", X.shape)
#     print("Shape of y: ", y.shape, y.dtype)
#     break


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# # Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),  # out: 6 * 24 * 24
            nn.ReLU(),
            # 2.估计是两层池化之后信息量丢失太大，导致准确率太低，试试移除池化层（准确率飙升！）
            nn.MaxPool2d(kernel_size=(2, 2)),  # out: 6 * 12 * 12
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # out: 16 * 8 * 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # out: 16 * 4 * 4
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            # nn.Linear(16 * 20 * 20, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 默认跑5遍
epochs = 5
# 3.试试多跑几遍（准确率提升7%）
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "data/model.pth")
print("Saved PyTorch Model State to model.pth")

# model = NeuralNetwork()
# model.load_state_dict(torch.load("data/model.pth"))
#
# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]
#
# model.eval()
# with torch.no_grad():
#     count = 0
#     correct = 0
#     for X, y in test_dataloader:
#         pred = model(X)
#         predicted, actual = classes[pred[0].argmax(0)], classes[y]
#         print(f'Predicted: "{predicted}", Actual: "{actual}"')
#         if predicted == actual:
#             correct += 1
#         count += 1
#         # 预测10个样本统计准确率
#         if count >= 10:
#             break
#     print(f'预测准确率：{100*correct/10}%')
