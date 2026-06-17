import torch
from tqdm import tqdm
from torch import nn
from collections import OrderedDict

from models.interface import Interface
from models.fasterrcnn.model import get_model
from utils.global_settings import BackdoorProperties

class FasterRCNNWrapper(Interface):
    def __init__(self, num_classes):
        # BOX_TRANSFORM_MODE is None unless --attack-type TARGETED_BOX_TRANSFORM was set,
        # so this is a no-op for every other configuration. Even when installed, the
        # backdoor RoI head only acts on batches whose targets carry a poison_mask
        # (i.e. malicious-dataset samples) and is never exercised at inference.
        self.model = get_model(
            model_path=None,
            num_classes=num_classes,
            box_transform_mode=BackdoorProperties.BOX_TRANSFORM_MODE,
            box_transform_params=BackdoorProperties.BOX_TRANSFORM_PARAMS,
        )
        self.num_classes = num_classes

    def get_weights(self):
        # Model is likely on the gpu when this function is called, if not no problem
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def get_fl_parameters(self):
        # Convert state_dict to list of numpy arrays as flower expects 
        state_dict = self.get_weights()
        return [val.numpy() for _, val in state_dict.items()]

    def set_weights(self, weights):
        """
        Load a torch state_dict
        weights: OrderedDict(Parameter: Weights)
        """
        self.model.load_state_dict(weights)

    def set_fl_parameters(self, parameters):
        """
        Convert the list of numpy arrays to state_dict that torch expects and loads them
        """
        keys = self.model.state_dict().keys()

        # State dict is torch order tensor dict, so convert in order to load the weights
        ordered_param_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})

        self.set_weights(ordered_param_dict)
    
    def get_model_state_keys(self):
        return self.get_weights().keys()

    @staticmethod
    def get_collate_fn():
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
        num_samples = len(dataloader)


        self.model.cpu()
        torch.cuda.empty_cache()
        return {"val_loss": avg_loss}, num_samples

    def train_step(self, dataloader, epochs, device, optimizer_config, partition_id=None, optimizer=None, grad_mask=None, loss_augment_fn=None, freeze_groups=None, epoch_callback=None):
        """
        Complete training loop for FasterRCNN
        Return: (weight tensor on CPU, number of samples, training metrics)

        grad_mask: optional dict[str, Tensor] keyed by named-parameter name. If supplied,
        each parameter's gradient is multiplied by its mask after backward and before the
        optimizer step (Neurotoxin-style projection onto the masked subspace).

        loss_augment_fn: optional callable (model -> scalar Tensor). When supplied, its
        return value is added to the FasterRCNN task losses before backward, so a single
        optimizer step minimizes both. Computed in fp32 OUTSIDE the autocast block — small
        anchor distances get rounded to zero in fp16 and a penalty would silently vanish.

        freeze_groups: optional iterable of named-parameter names to freeze for the
        duration of this train_step. Reserved for LaySelFL CT; raises NotImplementedError
        when set until the LaySelFL orchestrator (Phase 2) implements the proper
        requires_grad=False + optimizer-rebuild path.

        epoch_callback: optional callable (epoch_idx, state_dict_cpu) -> None invoked at the
        end of every epoch. The state dict is a CPU-side clone so the caller can stash many
        epoch snapshots without GPU memory pressure. Used by LaySelFL SLI's per-epoch L1
        scoring and by Pilot A's retroactive selection logs.
        """
        if freeze_groups is not None:
            raise NotImplementedError(
                "freeze_groups is reserved for LaySelFL CT (alternating sensitive-subset "
                "training with non-subset params frozen). Implementation lands with the "
                "LaySelFL orchestrator in Phase 2."
            )

        # Move model to gpu and put it in train mode
        self.model.to(device)
        self.model.train()

        # Move mask to device once so per-batch application is a cheap elementwise op
        device_grad_mask = None
        if grad_mask is not None:
            device_grad_mask = {n: m.to(device, non_blocking=True) for n, m in grad_mask.items()}

        if optimizer is None:
            # Optimizer setup
            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            # Old ugly way, but works for this specific case
            # If problems arise, move modern param_group construction
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
        scaler = torch.cuda.amp.GradScaler()

        final_epoch_losses = {}

        partition_text = ""
        if partition_id is not None:
            partition_text = f"[CLIENT {partition_id}] "

        # Dumb training loop
        for epoch in range(epochs):

            # Use tqdm to monitor training progress
            pbar = tqdm(
                dataloader, desc=f"{partition_text}Local Epoch {epoch + 1}/{epochs}", leave=True
            )
            
            # Accumulator for the batch losses of an epoch, used to compute epoch losses
            batch_losses = {}

            # Iterate through a batch
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
                # Penalty term lives OUTSIDE autocast. Anchor distances are typically
                # tiny (||theta - anchor||^2 from a few SGD steps of divergence); fp16
                # rounds them to zero and the penalty silently disappears. Cast losses
                # to fp32 before adding so the sum stays in full precision; autograd
                # handles the fp32->fp16 conversion at the AMP boundary on its own.
                if loss_augment_fn is not None:
                    losses = losses.float() + loss_augment_fn(self.model)

                # Backward pass, reset gradients
                optimizer.zero_grad()

                if scaler:
                    # Perform backward pass and calculate gradients
                    scaler.scale(losses).backward()
                    # Apply Neurotoxin gradient mask if supplied. A binary mask is
                    # invariant to AMP's scale factor, so no scaler.unscale_ is needed.
                    if device_grad_mask is not None:
                        for name, p in self.model.named_parameters():
                            if p.grad is not None and name in device_grad_mask:
                                p.grad.mul_(device_grad_mask[name])
                    # Let optimizer adjust model params
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    if device_grad_mask is not None:
                        for name, p in self.model.named_parameters():
                            if p.grad is not None and name in device_grad_mask:
                                p.grad.mul_(device_grad_mask[name])
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

            # Average all tracked loss components
            if epoch == epochs - 1:
                final_epoch_losses = {
                    k: sum(v_list) / len(v_list) for k, v_list in batch_losses.items()
                }
                # Rename to remain consistent with logger implementation, ugly
                final_epoch_losses["train_loss"] = final_epoch_losses.pop("total_loss")

            # Per-epoch snapshot hook. CPU-cloned so orchestrators (LaySelFL SLI, Pilot A)
            # can keep many epoch states across N epochs without GPU memory pressure.
            # Fires AFTER the batch loop but BEFORE moving the model off GPU so
            # state_dict() reads from the live GPU model on every epoch consistently.
            if epoch_callback is not None:
                epoch_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epoch_callback(epoch, epoch_state)


        self.model.cpu()
        torch.cuda.empty_cache()
        # Return: (Weights op CPU, Aantal Samples, Metrics. Optimizer(to keep momentum))
        return (
            self.get_weights(),
            len(dataloader.dataset),
            final_epoch_losses,
            optimizer
        )

    def evaluate_map_asr(self, clean_dl, poison_dl, device):
        pass
