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
        if logger is None:
            raise Exception("No server side logger provided for evaluation, can't save model")

        # Load new weights
        server_wrapper.set_fl_parameters(parameters)

        # Perform evaluation step
        metrics, num_samples = server_wrapper.evaluate(dataloader=val_dl, device=device)

        # Log the metrics
        logger.log_epoch(server_round, val_metrics=metrics)

        # Save the model using the logger
        weights = server_wrapper.get_weights()
        logger.save_if_best_model(epoch=server_round, val_loss=metrics["val_loss"], weights=weights)
        
        # Return the values that flower expects
        return float(metrics["val_loss"]), metrics

    return evaluate
