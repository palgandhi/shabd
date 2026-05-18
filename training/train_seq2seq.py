# training/train_seq2seq.py
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import model architectures
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.seq2seq import Seq2Seq, Encoder, Decoder

class TranslitDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

def get_collate_fn(pad_idx=0):
    def collate_fn(batch):
        src_list, tgt_list = [], []
        for src, tgt in batch:
            src_list.append(torch.tensor(src, dtype=torch.long))
            tgt_list.append(torch.tensor(tgt, dtype=torch.long))
            
        src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded
    return collate_fn

def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio, device, epoch_idx=1):
    model.train()
    epoch_loss = 0
    
    # Wrap dataloader in a beautiful tqdm progress bar
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx:02} Training", leave=False)
    
    for src, trg in pbar:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: [batch_size, trg_len, vocab_size]
        outputs = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
        
        # Slice predictions and targets to align outputs[:, t] -> trg[:, t+1]
        output_dim = outputs.shape[-1]
        predictions = outputs[:, :-1, :].reshape(-1, output_dim)
        targets = trg[:, 1:].reshape(-1)
        
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        
        # Show live batch loss in progress bar
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            
            # Turn off teacher forcing during evaluation
            outputs = model(src, trg, teacher_forcing_ratio=0.0)
            
            output_dim = outputs.shape[-1]
            predictions = outputs[:, :-1, :].reshape(-1, output_dim)
            targets = trg[:, 1:].reshape(-1)
            
            loss = criterion(predictions, targets)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def train_main():
    data_dir = "data"
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    train_path = os.path.join(data_dir, "train.pkl")
    dev_path = os.path.join(data_dir, "dev.pkl")
    
    # 1. Validation check
    if not (os.path.exists(vocab_path) and os.path.exists(train_path) and os.path.exists(dev_path)):
        print("Error: Vectorized dataset files not found. Please clean dataset via preprocess.py first.")
        return
        
    # 2. Load serialized dataset
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
        src_vocab = vocab_data["src_vocab"]
        tgt_vocab = vocab_data["tgt_vocab"]
        
    with open(train_path, "rb") as f:
        train_list = pickle.load(f)
    with open(dev_path, "rb") as f:
        dev_list = pickle.load(f)
        
    # 3. Dynamic device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Device designated: {device}")
    
    # 4. Instantiate data loaders
    batch_size = 64
    pad_idx = src_vocab.get('<PAD>', 0)
    collate = get_collate_fn(pad_idx=pad_idx)
    
    train_loader = DataLoader(TranslitDataset(train_list), batch_size=batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(TranslitDataset(dev_list), batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    # 5. Define Model Architectures
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 256
    DEC_HIDDEN_DIM = 512
    EPOCHS = 30
    PATIENCE = 5  # Stop if val loss doesn't improve for this many consecutive epochs
    
    encoder = Encoder(
        input_dim=len(src_vocab),
        emb_dim=EMBEDDING_DIM,
        enc_hid_dim=HIDDEN_DIM,
        dec_hid_dim=DEC_HIDDEN_DIM,
        dropout=0.3
    )
    
    decoder = Decoder(
        output_dim=len(tgt_vocab),
        emb_dim=EMBEDDING_DIM,
        dec_hid_dim=DEC_HIDDEN_DIM,
        dropout=0.3
    )
    
    sos_idx = tgt_vocab.get('<SOS>', 1)
    model = Seq2Seq(encoder, decoder, device, sos_idx=sos_idx).to(device)
    
    # 6. Optimizer & Loss Setup
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0          # early-stopping counter
    model_save_path = "models/seq2seq_best.pt"
    os.makedirs("models", exist_ok=True)
    os.makedirs("training/artifacts", exist_ok=True)
    
    log_file_path = "training/artifacts/training_log.txt"
    log_file = open(log_file_path, "a")
    log_file.write("\n--- Seq2Seq Training Log ---\n")
    
    print(f"Starting Seq2Seq training (Model 2) — up to {EPOCHS} epochs, patience={PATIENCE}...")
    teacher_forcing = 0.5
    
    for epoch in range(EPOCHS):
        # Gradual teacher forcing decay to ease generalization
        if epoch > 15:
            teacher_forcing = 0.3
            
        train_loss = train_epoch(model, train_loader, optimizer, criterion, teacher_forcing, device, epoch_idx=epoch+1)
        valid_loss = evaluate(model, dev_loader, criterion, device)
        
        scheduler.step(valid_loss)
        
        log_str = f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}"
        
        # Save checkpoint only when val loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            log_str += "  ✓ checkpoint saved"
        else:
            epochs_no_improve += 1
            log_str += f"  (no improvement {epochs_no_improve}/{PATIENCE})"
        
        print(log_str)
        log_file.write(log_str + "\n")
        log_file.flush()
        
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered after {epoch+1} epochs (patience={PATIENCE} exhausted).")
            print(f"  Best validation loss: {best_valid_loss:.4f}")
            log_file.write(f"Early stopping at epoch {epoch+1}. Best val loss: {best_valid_loss:.4f}\n")
            break
            
    log_file.close()
    print("Model 2 Seq2Seq training process completed!")

if __name__ == "__main__":
    train_main()
