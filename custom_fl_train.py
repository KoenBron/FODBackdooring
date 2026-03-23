from utils.parse_args import parse_args, apply_settings
from FL.custom_server import Server
from FL.custom_client import ClientManager


def start_simulation():
    """
    Main body for running the simultation
    """
    pass

if __name__ == "__main__":
    args = parse_args()
    apply_settings(args=args)
    start_simulation()


