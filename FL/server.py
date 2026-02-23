####################################
# Server evaluation factory function 
####################################
def get_evaluate_fn(server_wrapper, val_dl, device, logger):
    """
    Returns callback function for server evaluation function, that needs the
    the arguments the norma evaluate_fn doesn't have
    """
    def evaluate(server_round, parameters, config):
        """
        Performs the evaluation in order with how flower expects the return values
        """
        # Load new weights
        server_wrapper.set_fl_parameters(parameters)

        # Perform evaluation step
        metrics, num_samples = server_wrapper.evaluate(dataloader=val_dl, device=device)

        # Log the metrics
