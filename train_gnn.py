import torch

from torch_geometric.nn import SAGEConv

from utils import(
    load_reviews,
    create_pyg_data
)

class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(
            384,
            128
        )
        self.conv2 = SAGEConv(
            128,
            64
        )
        self.conv3 = SAGEConv(
            64,
            2
        )

    def forward(
            self,
            x,
            edge_index
    ):
        x = self.conv1(
            x,
            edge_index
        )
        x = torch.relu(x)
        x = self.conv2(
            x,
            edge_index
        )           
        x = torch.relu(x)
        x = self.conv3(
            x,
            edge_index
        )
        return x

def train():
    df = load_reviews(
        "reviews.csv"
    )
    data = create_pyg_data(df)
    model = FraudGNN()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )
    criterion = (
        torch.nn.CrossEntropyLoss()
    )
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(
            data.x,
            data.edge_index
        )
        loss = criterion(
            output,
            data.y
        )
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            pred = output.argmax(dim=1)
            accuracy = (
                pred == data.y
            ).sum().item() / len(data.y)
            print(
                f"Epoch {epoch} | "
                f"Loss {loss:.4f} | "
                f"Accuracy {accuracy:.2f}"
            )
    torch.save(
        model.state_dict(),
        "fraud_gnn.pt"
    )
    print(
        "Model saved as fraud_gnn.pt"
    )
if __name__ == "__main__":
    train()                