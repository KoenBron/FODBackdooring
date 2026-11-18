# models/

Model implementations behind a common wrapper interface, so the FL code never depends on a
specific architecture.

| File | Purpose |
|---|---|
| `interface.py` | The abstract base class defining how the FL code interacts with any model (weight get/set, local training, inference, evaluation). |
| `fasterrcnn/model.py` | Builds the torchvision `fasterrcnn_resnet50_fpn` detector. |
| `fasterrcnn/wrapper.py` | Implements the interface for Faster R-CNN — weight transfer, local training steps, inference. |
| `fasterrcnn/backdoor_roi_heads.py` | ROI-head surgery for the box-transform (regression-head) backdoor: rewrites box-regression targets for trigger-bearing objects during training. |
