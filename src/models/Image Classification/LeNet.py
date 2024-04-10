import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 첫 번째 컨볼루션 레이어: 1개의 입력 채널(흑백 이미지), 6개의 출력 채널, 5x5 컨볼루션
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 두 번째 컨볼루션 레이어: 6개의 입력 채널, 16개의 출력 채널, 5x5 컨볼루션
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 완전 연결 레이어: 16 * 5 * 5 입력 뉴런, 120 출력 뉴런 (5x5는 이미지 차원)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 완전 연결 레이어: 120 입력 뉴런, 84 출력 뉴런
        self.fc2 = nn.Linear(120, 84)
        # 완전 연결 레이어: 84 입력 뉴런, 10 출력 뉴런 (클래스 수)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 입력 x를 통해 순전파
        # 첫 번째 컨볼루션 레이어 후 ReLU 활성화 함수 적용
        x = F.relu(self.conv1(x))
        # 첫 번째 풀링 레이어 (2x2 서브샘플링)
        x = F.max_pool2d(x, 2)
        # 두 번째 컨볼루션 레이어 후 ReLU 활성화 함수 적용
        x = F.relu(self.conv2(x))
        # 두 번째 풀링 레이어 (2x2 서브샘플링)
        x = F.max_pool2d(x, 2)
        # Flatten
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 마지막 완전 연결 레이어
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # MNIST 트레이닝 데이터셋
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # MNIST 테스트 데이터셋
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, epochs=1):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    model = LeNet().to(device)

    print(summary(model, input_size=(1, 32, 32)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    batch_size = 64
    trainloader, testloader = get_mnist_data(batch_size)

    train_model(model, trainloader, criterion, optimizer, epochs=1)
