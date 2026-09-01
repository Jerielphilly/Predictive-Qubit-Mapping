import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


print("Loading GNN dataset...")


dataset = torch.load(
    "gnn_dataset.pt",
    weights_only=False
)


print(
    "Total graph samples:",
    len(dataset)
)


train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42
)


print(
    "Training samples:",
    len(train_data)
)

print(
    "Testing samples:",
    len(test_data)
)


train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)


test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)


class GNNRouter(nn.Module):

    def __init__(self):

        super().__init__()


        self.conv1 = GCNConv(
            9,
            32
        )


        self.conv2 = GCNConv(
            32,
            32
        )


        self.conv3 = GCNConv(
            32,
            16
        )


        self.fc1 = nn.Linear(
            16,
            16
        )


        self.fc2 = nn.Linear(
            16,
            1
        )


        self.relu = nn.ReLU()


    def forward(
        self,
        x,
        edge_index,
        batch
    ):


        x = self.conv1(
            x,
            edge_index
        )

        x = self.relu(x)


        x = self.conv2(
            x,
            edge_index
        )

        x = self.relu(x)


        x = self.conv3(
            x,
            edge_index
        )

        x = self.relu(x)


        x = global_mean_pool(
            x,
            batch
        )


        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)


        return x.squeeze(-1)


device = torch.device(
    "cpu"
)


model = GNNRouter().to(
    device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


loss_function = nn.MSELoss()


EPOCHS = 30


print()
print("========================================")
print("          TRAINING GNN MODEL")
print("========================================")


for epoch in range(EPOCHS):


    model.train()


    total_loss = 0


    for batch in train_loader:


        batch = batch.to(
            device
        )


        optimizer.zero_grad()


        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )


        targets = batch.y.view(
            -1
        )


        loss = loss_function(
            predictions,
            targets
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
            * batch.num_graphs
        )


    average_loss = (
        total_loss
        / len(train_data)
    )


    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} "
        f"Loss: {average_loss:.4f}"
    )


print()
print("Training complete!")


model.eval()


actual = []
predicted = []


with torch.no_grad():


    for batch in test_loader:


        batch = batch.to(
            device
        )


        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )


        targets = batch.y.view(
            -1
        )


        actual.extend(
            targets.cpu().numpy()
        )


        predicted.extend(
            predictions.cpu().numpy()
        )


mae = mean_absolute_error(
    actual,
    predicted
)


mse = mean_squared_error(
    actual,
    predicted
)


r2 = r2_score(
    actual,
    predicted
)


print()
print("========================================")
print("          GNN MODEL RESULTS")
print("========================================")


print()

print(
    "Training samples:",
    len(train_data)
)


print(
    "Testing samples:",
    len(test_data)
)


print(
    "MAE:",
    round(mae, 4)
)


print(
    "MSE:",
    round(mse, 4)
)


print(
    "R2 Score:",
    round(r2, 4)
)


torch.save(
    model.state_dict(),
    "gnn_routing_model.pt"
)


print()
print(
    "Model saved as gnn_routing_model.pt"
)

print("========================================")