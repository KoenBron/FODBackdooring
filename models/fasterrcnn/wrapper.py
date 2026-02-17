import torch
from models.interface import Interface
from models.fasterrcnn.model import get_model
from tqdm import tqdm

class FasterRCNNWrapper(Interface):
    def __init__(self, num_classes):
        self.model = get_model(model_path=None, num_classes=num_classes)
        self.num_classes = num_classes

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_collate_fn(self):
        def collate_fn(batch):
            return tuple(zip(*batch))
        return collate_fn

    def train_step(self, dataloader, epochs, device, optimizer_config):
        """
        Complete training loop for FasterRCNN
        Return: (weight tensor on CPU, number of samples, training metrics)
        """
        # Move model to gpu and put it in train mode
        self.model.to(device)
        self.model.train()

        # Optimizer setup
        params = [p for p in self.model.parameters() if p.requires_grad]
        # TODO: Add parameter groups back to adjust weight decay per parameter group like in train.py of centralized fasterrcnn  implementation
        optimizer = torch.optim.SGD(
            params, 
            lr=optimizer_config.get("lr", 0.005),
            momentum=optimizer_config.get("momentum", 0.9),
            weight_decay=optimizer_config.get("weight_decay", 0.0005)
        )

        # Scaler for Mixed Precision
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        epoch_losses = []

        for epoch in range(epochs):
            batch_losses = []
            
            # Use tqdm to monitor training progress
            pbar = tqdm(dataloader, desc=f"Local Epoch {epoch+1}/{epochs}", leave=False)
            
            for images, targets in pbar:
                # Images and targets are already batched as a tuple through custom_collate function
                # Upload both to the gpu, use non_blocking for asynchronisity
                images = [img.to(device, non_blocking=True) for img in images]# List of tensors
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]# List of dict-tensors

                # Forward pass with auto scaling 
                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # Backward pass, reset gradients
                optimizer.zero_grad()
                
                if scaler:
                    # Perform backward pass and calculate gradients
                    scaler.scale(losses).backward()
                    # Let optimizer adjust model params
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()

                # Bookkeeping
                loss_val = losses.item()
                batch_losses.append(loss_val)
                pbar.set_postfix({'loss': f"{loss_val:.4f}"})

            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        # Return: (Weights op CPU, Aantal Samples, Metrics)
        return self.get_weights(), len(dataloader.dataset), {"train_loss": epoch_losses[-1]}
