from torch import nn, Tensor, optim
from torch.utils.data import DataLoader



class BookieModel(nn.Module):
    def __init__(self, input_len: int, output_len: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_len, 40),
            nn.ReLU(),
            nn.Linear(40, output_len),
            nn.Tanh()
        )

    def forward(self, input: int):
        return self.layers(input)
    
def train_model(model: BookieModel, loader: DataLoader, epochs: int, learning_rate: float) -> Tensor:
    """
    train_data.shape: [length, width]
    Recommended:
    learning_rate: 0.1-0.0001
    epochs: 20-40

    returns loss of last epoch
    """

    loss_fn = nn.MSELoss()
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    model.train()

    loss = Tensor([-1])

    for epoch in range(epochs):
        print("\nEpoch:", epoch)
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print("Loss:", loss)

    return loss
