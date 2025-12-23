import logging
import os
import sys

def setup_logger(name, log_dir, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
