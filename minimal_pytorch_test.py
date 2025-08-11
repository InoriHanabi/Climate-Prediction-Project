# minimal_pytorch_test.py
import torch
import torch.nn as nn

DEVICE = torch.device("cpu") # Ensure it's CPU

# Dummy data shapes similar to your climate task
BATCH_SIZE = 16
WINDOW_SIZE = 12 # Or 30 like daily, doesn't matter much for this test
INPUT_FEATURES = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OFFSET = 12 # Or 1 like daily

# Simplified Model (matching your LSTMForecaster structure)
class MinimalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden_state = hn[-1] 
        out = self.fc(last_hidden_state)
        return out

model = MinimalLSTM(INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, OFFSET).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting minimal PyTorch CPU training loop test...")
try:
    for epoch in range(5): 
        x_batch = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_FEATURES).to(DEVICE)
        y_batch = torch.randn(BATCH_SIZE, OFFSET).to(DEVICE) 

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print("Minimal test completed successfully.")
except Exception as e:
    print(f"Error in minimal test: {e}")