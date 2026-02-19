import torch
import argparse
from torch.utils.data import DataLoader

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, TrainingSettings
from utils.model_log import ModelLogger


def run_centralized_training(args):
    print("-------------Starting Centralize Training-----------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "always_clean"# We create a clean model

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

    logger = ModelLogger()

    # Training loop
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
        logger.log_epoch(epoch, train_metrics=train_metrics, val_metrics=val_metrics)
        logger.save_if_best_model(epoch=epoch, val_loss=val_metrics['val_loss'], weights=weights, optimizer=optimizer)

    logger.close()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    run_centralized_training(args)
