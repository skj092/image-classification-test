# defining the training function 
from tqdm import tqdm
import torch



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in tqdm(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            yb_ = model(xb)
            loss = criterion(yb_, yb)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append((yb_.argmax(1) == yb).float().mean())
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(val_loader):
                xb, yb = xb.to(device), yb.to(device)
                yb_ = model(xb)
                loss = criterion(yb_, yb)
                val_loss.append(loss.item())
                val_acc.append((yb_.argmax(1) == yb).float().mean())
        print(f'Epoch: {epoch+1}, Train Loss: {torch.tensor(train_loss).mean():.4f}, Train Accuracy: {torch.tensor(train_acc).mean():.4f}, Val Loss: {torch.tensor(val_loss).mean():.4f}, Val Accuracy: {torch.tensor(val_acc).mean():.4f}')
        
# Inference 

# defining the inference function
def inference(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, yb in tqdm(test_loader):
            xb = xb.to(device)
            yb_ = model(xb)
            predictions.extend(yb_.argmax(1).cpu().numpy())
    return predictions
