import logging

def configure_logging(log_level=logging.INFO, log_file=None):
    log_level = logging.getLevelName(log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)] if log_file else [logging.StreamHandler()],
    )
