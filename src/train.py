import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import time
import glob
import random
import gc

# Import custom modules
from src.dataset import GoDataset
from src.model import GoResNet

# --- CONFIG ---
BATCH_SIZE = 256        # 256 Boards per batch
LEARNING_RATE = 0.001   # Adam LR
EPOCHS = 3              # Total epochs to train
CHUNK_SIZE = 15000      # Games per chunk load

def save_graph(losses, epoch_num, is_final=False):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    
    if is_final:
        title = "Final Convergence (All Data)"
        path = "assets/training_graph_final.png"
    else:
        title = f"Convergence - Epoch {epoch_num}"
        path = f"assets/training_graph_epoch_{epoch_num}.png"
        
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(path)
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Starting Chunked Training on {device}...")
    
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("assets"): os.makedirs("assets")

    # 1. Get List of ALL Files
    all_files = glob.glob(os.path.join('data/processed', '*.npz'))
    total_files = len(all_files)
    print(f" Found {total_files} total games on disk.")
    
    if total_files == 0:
        print(" No processed data found! Run src/process.py first.")
        return

    # Initialize Model
    model = GoResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    start_time = time.time()

    # 2. Main Training Loop
    for epoch in range(EPOCHS):
        print(f"\n=== STARTING EPOCH {epoch+1}/{EPOCHS} ===")
        
        # Shuffle file list so chunks are random every time
        random.shuffle(all_files)
        
        # Create Chunks
        chunks = [all_files[i:i + CHUNK_SIZE] for i in range(0, total_files, CHUNK_SIZE)]
        total_chunks = len(chunks)
        
        epoch_loss_sum = 0
        total_batches = 0

        # 3. Chunk Loop (Load -> Train -> Dump)
        for chunk_idx, file_chunk in enumerate(chunks):
            print(f" Loading Chunk {chunk_idx+1}/{total_chunks} ({len(file_chunk)} games)...")
            
            # --- FIX: Pass the file list directly ---
            dataset = GoDataset(file_chunk) 
            
            if len(dataset) == 0: 
                print("      ⚠️ Chunk empty, skipping.")
                continue
            
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
            
            model.train()
            
            # Batch Loop
            for i, (boards, moves) in enumerate(dataloader):
                boards, moves = boards.to(device), moves.to(device)
                
                optimizer.zero_grad()
                outputs = model(boards)
                loss = criterion(outputs, moves)
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                epoch_loss_sum += loss_val
                train_losses.append(loss_val)
                
                # Logging
                if i % 500 == 0:
                    print(f"      Batch {i}: Loss {loss_val:.4f}")

            # Cleanup RAM
            total_batches += len(dataloader)
            del dataset
            del dataloader
            gc.collect() # Force garbage collection
            print(f"      ✅ Chunk {chunk_idx+1} Finished. RAM cleared.")

        # End of Epoch
        avg_loss = epoch_loss_sum / total_batches if total_batches > 0 else 0
        print(f" Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"models/resnet_go_epoch_{epoch+1}.pth")
        save_graph(train_losses, epoch+1)

    # Final Save
    print(" Saving Final Model...")
    torch.save(model.state_dict(), "models/resnet_go_final.pth")
    save_graph(train_losses, EPOCHS, is_final=True)
    
    # Save Stats
    elapsed = (time.time() - start_time) / 60
    stats = {
        "total_games_available": total_files,
        "chunk_size": CHUNK_SIZE,
        "final_loss": train_losses[-1] if train_losses else 0,
        "training_time_min": round(elapsed, 1)
    }
    with open("assets/model_stats.json", "w") as f:
        json.dump(stats, f)

    print(f" Training Finished in {elapsed:.1f} minutes.")

if __name__ == "__main__":
    train()