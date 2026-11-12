# Folder Purpose
Model implementations, each behind a common wrapper interface so the FL code never depends on a
specific architecture. `interface.py` at the top of this directory defines the abstract base
class every wrapper implements.

## Folder Structure
- `interface.py`: Abstract base class defining how the FL code interacts with a model.
- `fasterrcnn/`:
    - `model.py`: Builds the torchvision `fasterrcnn_resnet50_fpn` model.
    - `wrapper.py`: Implements the interface — weight get/set, local training steps, inference.
    - `backdoor_roi_heads.py`: ROI-head surgery used by the box-transform (regression-head)
      backdoor to rewrite box-regression targets for trigger-bearing objects.
