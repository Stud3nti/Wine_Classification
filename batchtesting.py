import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

dataset = datasets.load_wine()
X, y = dataset.data, dataset.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

mean = torch.mean(X, dim=0)
std = torch.std(X, dim=0)
X = (X - mean) / std

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=4)
test_loader = DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=4)




class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.softmax(out, dim=1)
        return out

model = LogisticRegression(input_size=13, num_classes=3)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Move the model to the device
model = model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

# Define training parameters
num_epochs = 1000

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training loss for each epoch
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))



with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        # Move inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Compute the model's predictions
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Compute the accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Validation Accuracy: {:.2f}%'.format(100 * correct / total))