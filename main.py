import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.FashionMNIST(root="data",
                                      train=True,
                                      download=True,
                                      transform=ToTensor(),
                                      target_transform=None)
test_dataset = datasets.FashionMNIST(root="data",
                                     train=False,
                                     download=True,
                                     transform=ToTensor(),
                                     target_transform=None)


class_names = train_dataset.classes

torch.manual_seed(0)
fig = plt.figure(figsize=(9, 9))
rows, cols = 2, 2
for i in range(1, rows * cols + 1):
  random_idx = torch.randint(0, len(train_dataset), size=[1]).item()
  img, label = train_dataset[random_idx]
  ax = fig.add_subplot(rows, cols, i)
  ax.imshow(img.squeeze(), cmap="gray")
  ax.set_title(class_names[label])
  ax.axis(False)
plt.show()
plt.close()

train_dataloader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False)


class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_features, hidden_layers, output_features):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_features, out_channels=hidden_layers, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # !! IMPORTANT CORRECTION HERE: The in_channels for the second Conv2d in conv_layer_2
            #    should be hidden_layers, not input_features, if you're chaining from the previous layer's output.
            nn.Conv2d(in_channels=hidden_layers, out_channels=hidden_layers, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_layers * 7 * 7, out_features=output_features)
        )

    def forward(self, x):
        # Access the layers using 'self.'
        x = self.conv_layer_1(x)
        print(x.shape)
        x = self.conv_layer_2(x)
        print(x.shape)
        x = self.classification(x)
        print(x.shape)
        return x # Return 'x' directly, not 'self.x'

torch.manual_seed(0)
model_2 = FashionMNISTModelV2(input_features=1, hidden_layers=10, output_features=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device):
  train_loss, train_acc = 0, 0
  model.train()
  for batch, (X_train, y_train) in enumerate(data_loader):
    X_train, y_train = X_train.to(device), y_train.to(device)

    y_preds = model(X_train)

    loss = loss_fn(y_preds, y_train)
    train_loss += loss

    train_acc += accuracy_fn(y_train, torch.softmax(y_preds, dim=1).argmax(dim=1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss  /= len(data_loader)
  train_acc /= len(data_loader)
  print(f"Train loss: {train_loss} | Train acc: {train_acc}")

def test_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device):
  test_loss, test_acc = 0, 0
  model.eval()

  with torch.inference_mode():
    for batch, (X_test, y_test) in enumerate(data_loader):
      X_test, y_test = X_test.to(device), y_test.to(device)

      y_preds = model(X_test)

      test_loss += loss_fn(y_preds, y_test)

      test_acc += accuracy_fn(y_test, torch.softmax(y_preds, dim=1).argmax(dim=1))

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Test loss: {test_loss} | Test acc: {test_acc}")

### Training begins
from timeit import default_timer as timer
from tqdm.auto import tqdm

train_start_time = timer()

torch.manual_seed(0)
epochs = 3

for epoch in tqdm(range(epochs)):
  print(f"--Epoch: {epoch} --\n")
  train_step(model=model_2, data_loader=train_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, optimizer=optimizer, device=device)
  test_step(model=model_2, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

train_end_time = timer()

print(f"The time passed for training on {device}: {train_end_time - train_start_time}")
