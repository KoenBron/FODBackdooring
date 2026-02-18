import torch
import os
import argparse
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, TrainingSettings


def run_centralized_training(args):
    print("-------------Starting Centralize Training-----------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = ProjectPaths.OUTPUT_DIR# Where we store the resulting model weights
    date_format = "%d-%m-%Y-%H-%M-%S"
    date_string = time.strftime(date_format)
    output_dir = os.path.join(exp_dir, date_string)
    writer = SummaryWriter(output_dir)

    mode = "always_clean"

    print("-------------Loading Datasets-----------------")
    # Create the datasets
    train_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR, # Of args.data_dir + /train/images
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
        mode=mode
    )
    
    val_ds = CustomCocoDataset(
        image_folder=ProjectPaths.VALIDATION_IMAGES_DIR, # Eigenlijk validation set paths hier
        annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
        mode=mode
    )

    # Initiate the wrapper to direct the FasterRCNN model 
    wrapper = FasterRCNNWrapper(num_classes=train_ds.get_total_classes_count() + 1)# Account for background class

    # Create the dataloaders that use the datasets
    train_dl = DataLoader(
        train_ds,
        batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=wrapper.get_collate_fn(),
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=TrainingSettings.TRAINING_BATCH_SIZE,
        shuffle=False,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=wrapper.get_collate_fn(),
        pin_memory=True,
    )
     
    # Recording the best valuation score
    best_val_loss = float('inf')
    optimizer = None # We bewaren de optimizer tussen epochs door!
    
    optimizer_config = {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # A. TRAIN (1 Epoch per keer aanroepen)
        # We geven 'optimizer' mee zodat momentum bewaard blijft!
        weights, _, train_metrics, optimizer = wrapper.train_step(
            dataloader=train_dl, 
            epochs=1, 
            device=device, 
            optimizer_config=optimizer_config,
            optimizer=optimizer 
        )
        
        # B. EVALUATE
        val_metrics = wrapper.evaluate(val_dl, device)
        
        # C. LOGGING
        print(f"Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        writer.add_scalar("Loss/Train", train_metrics['train_loss'], epoch)
        writer.add_scalar("Loss/Val", val_metrics['val_loss'], epoch)
        
        # D. SAVE BEST MODEL
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_path = os.path.join(output_dir, "best_model.pth")
            
            os.makedirs(output_dir, exist_ok=True)
            # We slaan de weights op (CPU-friendly)
            # En optioneel de optimizer state als je wilt kunnen resumen
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': wrapper.get_weights(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, save_path)
            print(f"✅ New Best Model Saved to {save_path}")

    writer.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    run_centralized_training(args)
