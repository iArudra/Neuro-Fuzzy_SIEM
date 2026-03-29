import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import NeuroFuzzySIEM

DATA_FILE = r"d:\College\PROJECTS\SIEM\processed_data.pkl"
MODEL_FILE = r"d:\College\PROJECTS\SIEM\neuro_fuzzy_siem.pth"
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
VAL_SPLIT = 0.2

def load_data():
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    print("Loading preprocessed data...")
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return
        
    data = load_data()
    X_cat = data['X_cat']
    X_num = data['X_num']
    y = data['y']
    vocab_size = data['vocab_size']
    
    # Shuffle and split
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split_idx = int(len(y) * (1 - VAL_SPLIT))
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_cat_train, X_num_train, y_train = X_cat[train_idx], X_num[train_idx], y[train_idx]
    X_cat_val, X_num_val, y_val = X_cat[val_idx], X_num[val_idx], y[val_idx]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.tensor(X_cat_train, dtype=torch.long),
        torch.tensor(X_num_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(X_num_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NeuroFuzzySIEM(vocab_size=vocab_size, num_numeric_features=X_num.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    mse_loss_fn = nn.MSELoss()
    
    train_losses = []
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for b_X_cat, b_X_num, b_y in train_loader:
            b_X_cat, b_X_num, b_y = b_X_cat.to(device), b_X_num.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            
            risk_scores = model(b_X_cat, b_X_num) # 0 to 100
            
            # BCE Loss
            prob = risk_scores / 100.0
            prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
            bce_loss = nn.BCELoss()(prob, b_y)
            
            # MSE Loss
            target_risk = b_y * 100.0
            mse_loss = mse_loss_fn(risk_scores, target_risk)
            
            loss = bce_loss + 0.3 * mse_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
    print("Saving model...")
    torch.save(model.state_dict(), MODEL_FILE)
    
    print("Evaluating on validation set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for b_X_cat, b_X_num, b_y in val_loader:
            b_X_cat, b_X_num = b_X_cat.to(device), b_X_num.to(device)
            risk_scores = model(b_X_cat, b_X_num)
            all_scores.extend(risk_scores.cpu().numpy())
            preds = (risk_scores > 50.0).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(b_y.cpu().numpy())
            
    # Metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    all_probs = all_scores / 100.0
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 
        
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"\n--- Validation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"FPR:       {fpr:.4f}")
    
    # Save plots
    try:
        plt.figure()
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(r'd:\College\PROJECTS\SIEM\loss_curve.png')
        plt.close()
        
        plt.figure()
        sns.histplot(all_scores, bins=20, kde=True)
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.savefig(r'd:\College\PROJECTS\SIEM\risk_score_dist.png')
        plt.close()
        
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(r'd:\College\PROJECTS\SIEM\confusion_matrix.png')
        plt.close()
        print("Plots saved.")
    except Exception as e:
        print(f"Error saving plots: {e}")

if __name__ == "__main__":
    main()
