import torch
import argparse
from torch.utils.data import DataLoader
import os

from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.global_settings import ProjectPaths, TrainingSettings
from utils.model_log import ModelLogger


def run_centralized_training(args):
    if args.output_dir is not None:
        ProjectPaths.OUTPUT_DIR = os.path.join(
            ProjectPaths.OUTPUT_DIR, "..", args.output_dir
        )
    print("-------------Starting Centralize Training-----------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-------------Loading Datasets-----------------")
    # Create the datasets
    train_ds = CustomCocoDataset(
        image_folder=ProjectPaths.TRAIN_IMAGES_DIR,  # Of args.data_dir + /train/images
        annotations_file=ProjectPaths.TRAIN_ANNOTATIONS_FILE,
        mode="training",
    )

    val_ds = CustomCocoDataset(
        image_folder=ProjectPaths.VALIDATION_IMAGES_DIR,  # Eigenlijk validation set paths hier
        annotations_file=ProjectPaths.VALIDATION_ANNOTATIONS_FILE,
        mode="always_clean",
    )

    # Initiate the wrapper to direct the FasterRCNN model
    wrapper = FasterRCNNWrapper(
        num_classes=train_ds.get_total_classes_count() + 1
    )  # Account for background class

    # Create the dataloaders that use the datasets
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=wrapper.get_collate_fn(),
        pin_memory=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TrainingSettings.NUM_WORKERS,
        collate_fn=wrapper.get_collate_fn(),
        pin_memory=True,
    )

    # Recording the best valuation score
    optimizer = None  # We bewaren de optimizer tussen epochs door!

    optimizer_config = {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}

    logger = ModelLogger()

    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Re-use optimizer to keep momentum
        weights, _, train_metrics, optimizer = wrapper.train_step(
            dataloader=train_dl,
            epochs=1,
            device=device,
            optimizer_config=optimizer_config,
            optimizer=optimizer,
        )

        val_metrics, _ = wrapper.evaluate(val_dl, device)

        print(
            f"Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}"
        )
        logger.log_epoch(epoch, train_metrics=train_metrics, val_metrics=val_metrics)
        logger.save_if_best_model(
            epoch=epoch,
            val_loss=val_metrics["val_loss"],
            weights=weights,
            optimizer=optimizer,
        )

    logger.close()
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The subdirectory to save the resulting model to in the output directory, default is output_dir/fl/",
    )
    args = parser.parse_args()

    run_centralized_training(args)
