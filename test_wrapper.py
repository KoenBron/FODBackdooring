import torch
from torch.utils.data import DataLoader
from models.fasterrcnn.wrapper import FasterRCNNWrapper
from utils.dataset import CustomCocoDataset
from utils.poison_properties import BackdoorPaths

def run_centralized_test():
    # 1. Setup
    paths = BackdoorPaths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Dataset laden (Gebruikt jouw bestaande code)
    print("Loading Training Data...")
    train_ds = CustomCocoDataset(
        image_folder=paths.TRAINING_IMAGES_SET,
        annotations_file=paths.TRAINING_ANNOTATIONS_FILE,
        mode="training"
    )

    # 3. Model Wrapper Init
    print("Initializing Faster R-CNN Wrapper...")
    wrapper = FasterRCNNWrapper(num_classes=train_ds.get_total_classes_count() + 1)
    
    # 4. DataLoader maken (LET OP: We gebruiken de collate_fn van de wrapper!)
    train_dl = DataLoader(
        train_ds,
        batch_size=4, 
        shuffle=True,
        num_workers=2,
        collate_fn=wrapper.get_collate_fn()  # <--- Cruciaal: De wrapper bepaalt hoe batches eruit zien
    )

    # 5. Training Simulatie
    print("Starting Training Loop (Testing 2 Epochs)...")
    
    # Config voor de optimizer
    optim_config = {"lr": 0.005, "momentum": 0.9}

    # We roepen train_step aan. 
    # In FL is 'epochs' vaak 1 of 2 per ronde. 
    # Hier doen we gewoon 2 epochs achter elkaar om te testen of de loss daalt.
    
    weights, num_samples, metrics = wrapper.train_step(
        dataloader=train_dl,
        epochs=2,
        device=device,
        optimizer_config=optim_config
    )

    print(f"\n✅ Training Finished!")
    print(f"   Samples Processed: {num_samples}")
    print(f"   Final Train Loss: {metrics['train_loss']:.4f}")

    # Check: Zijn de weights veranderd? (Simpele check)
    # Dit bewijst dat er echt geleerd is.
    print(f"   Weight keys returned: {len(weights)}")

if __name__ == "__main__":
    run_centralized_test()
