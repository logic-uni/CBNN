import torch

import torch.nn as nn
import torch.optim as optim

class TimeSeriesMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_interval):
        super(TimeSeriesMLP, self).__init__()
        self.time_interval = time_interval
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def set_time_interval(self, new_interval):
        self.time_interval = new_interval

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
time_interval = 5

model = TimeSeriesMLP(input_size, hidden_size, output_size, time_interval)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy input and target
input_data = torch.randn(1, input_size)
target = torch.randn(1, output_size)

# Training step
model.train()
optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# Modify time interval
model.set_time_interval(10)