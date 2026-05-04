# Folder Purpose
This directory contains the implementations of different models organised in subdirectories.
Each model has its own model specific pytorch implementation and an accompanying interface.py file that defines how you can interact with the model. 
The interface.py file in the top of this directory contains the abstract base class for this interface that all of the models should adhere to.

## Folder Structure
- interface.py: Holds the abstract base class defining model interactions
- fasterrcnn/:
    - model.py: creates the model
    - wrapper.py: Defines interactions with the model
    
