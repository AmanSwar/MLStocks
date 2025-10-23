
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os..dirname(os.path.abspath(__file__))))

from data_prep.data_load import prepare_data
from data_prep.data_clean import clean_indicator
from model.xlstm import MLSTMBlock
from torch.utils.data import TensorDataset, DataLoader

# --- Configuration ---
DATA_PATH = "/home/aman/code/ml_fr/ml_stocks/data/NIFTY_5_years.csv"
OUTPUT_DIR = "/home/aman/code/ml_fr/ml_stocks/output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")
LOSS_PLOT_FILE = os.path.join(OUTPUT_DIR, "training_loss.png")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "xlstm_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Preparing data...")
df = prepare_data(DATA_PATH)
df = clean_indicator(df)

features = df.drop(columns=['Daily_Return']).values
labels = df['Daily_Return'].values

features = features[:-1]
labels = labels[1:]

X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("Data preparation complete.")

print("Initializing model...")


input_size = X.shape[1]
hidden_size = 128
num_heads = 4
model = MLSTMBlock(hidden_size=hidden_size, num_heads=num_heads)

class xLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(xLSTMModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.xlstm_block = MLSTMBlock(hidden_size=hidden_size, num_heads=num_heads)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, state):
        x = self.embedding(x)
        x, new_state = self.xlstm_block(x, state)
        x = self.fc_out(x)
        return x, new_state

    def init_state(self, batch_size, device):
        return self.xlstm_block.init_state(batch_size, device)

model = xLSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=1, num_heads=num_heads)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print("Model initialization complete.")

print("Starting training...")
num_epochs = 20
losses = []
log_data = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        model.train()
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        state = model.init_state(batch_X.size(0), device)

        optimizer.zero_grad()

        output, _ = model(batch_X, state)

        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    losses.append(avg_epoch_loss)
    log_data.append({"epoch": epoch + 1, "loss": avg_epoch_loss})
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

print("Training complete.")

print("Logging training data...")
log_df = pd.DataFrame(log_data)
log_df.to_csv(LOG_FILE, index=False)
print(f"Training log saved to {LOG_FILE}")

print("Generating loss plot...")
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(LOSS_PLOT_FILE)
plt.close()
print(f"Loss plot saved to {LOSS_PLOT_FILE}")

print("Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

print("Script finished successfully.")
