# Folder Purpose
Holds the Federate Learning specific code. **Note**: `client.py` & `server.py` are legacy implementations that used the flwr package to simulate FL. `custom_client.py` & `custom_server.py` hold the current custom implementations

## Folder Structure:
- strategies/:
    - fedavg.py: Implementation of the federated averaging strategy for client update aggregation
- custom_client.py: Client manager implementation, active clients are configurations loaded by the  manager at that point making this clients statefull and easily swappable.
- custom_server.py: Server implemenation responsible for aggregating and averaging client updates and also logging of progress using the model_log.py functionalities.
