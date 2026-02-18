import torch
from tqdm import tqdm
from torch import nn

from models.interface import Interface
from models.fasterrcnn.model import get_model

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

    def evaluate(self, dataloader, device):
        """
        Runs a single evaluation epoch of the loss on the validation dataset and returns the loss
        """
        # Set to training mode to have the model output be a loss dict instead of bboxes
        self.model.train()
        self.model.to(device)

        epoch_loss = 0
        steps = 0

        # Disable gradient computation
        with torch.no_grad():
            bar = tqdm(dataloader, desc="Evaluating validation set", leave=False)
            for images, targets in bar:
                # Get the images and targets in the batch as lists of tensors
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in target.items()} for target in targets
                ]

                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # Get loss dict from the gpu to cpu
                epoch_loss += losses.item()
                steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else float("inf")
        return {"val_loss": avg_loss}

    def train_step(self, dataloader, epochs, device, optimizer_config, optimizer=None):
        """
        Complete training loop for FasterRCNN
        Return: (weight tensor on CPU, number of samples, training metrics)
        """
        # Move model to gpu and put it in train mode
        self.model.to(device)
        self.model.train()

        if optimizer is None:
            # Optimizer setup
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            # In case of problems, see if this is handles appropriately
            # for name, param in self.model.named_parameters():
            #     if not param.requires_grad:
            #         continue  # Skip non-trainable layers
            #
            #     if "bias" in name:
            #         pg2.append(param)  # Biases (no decay)
            #     elif "bn" in name or "norm" in name:
            #         pg0.append(param)  # Batchnorm weights (no decay)
            #     else:
            #         pg1.append(param)  # Weights (decay)

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            # Set params for each param_group in the optimizer, here batchnorm params
            optimizer = torch.optim.SGD(
                pg0,
                lr=optimizer_config.get("lr", 0.005),
                momentum=optimizer_config.get("momentum", 0.9),
                nesterov=True,
            )

            optimizer.add_param_group(
                {
                    "params": pg1,
                    "weight_decay": optimizer_config.get("weight_decay", 5e-4),
                }
            )  # Weights with weigth decay
            optimizer.add_param_group({"params": pg2})  # Biases

        # Scaler for Mixed Precision
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        final_epoch_losses = {}

        for epoch in range(epochs):

            # Use tqdm to monitor training progress
            pbar = tqdm(
                dataloader, desc=f"Local Epoch {epoch + 1}/{epochs}", leave=False
            )
            
            # Accumulator for the batch losses of an epoch, used to compute epoch losses
            batch_losses = {}

            for images, targets in pbar:
                # Images and targets are already batched as a tuple through custom_collate function
                # Upload both to the gpu, use non_blocking for asynchronisity
                images = [
                    img.to(device, non_blocking=True) for img in images
                ]  # List of tensors
                targets = [
                    {k: v.to(device, non_blocking=True) for k, v in t.items()}
                    for t in targets
                ]  # List of dict-tensors

                # Forward pass with auto scaling
                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())# Total Losses

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

                # Keep track of the metrics
                loss_val = losses.item()# Total losses to cpu
                
                # Total batch loss
                if "total_loss" not in batch_losses.keys():
                    batch_losses["total_loss"] = []
                batch_losses["total_loss"].append(loss_val)

                # Loss components
                for k, v in loss_dict.items():
                    if k not in batch_losses:
                        batch_losses[k] = []
                    batch_losses[k].append(v.item())

                # Bookkeeping of progressbar
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Reset epoch losses
            if epoch == epochs - 1:
                final_epoch_losses = {
                    k: sum(v_list) / len(v_list) for k, v_list in batch_losses.items()
                }
                final_epoch_losses["train_loss"] = final_epoch_losses.pop("total_loss")

        # Return: (Weights op CPU, Aantal Samples, Metrics)
        return (
            self.get_weights(),
            len(dataloader.dataset),
            final_epoch_losses,
            optimizer
        )
