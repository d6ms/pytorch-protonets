import os

def create_dirs(base):
    if len(base) == 0:
        base = '.'
    log_dir = f'{base}/logs'
    model_dir = f'{base}/models'
    data_dir = f'{base}/data'
    for dir in [log_dir, model_dir, data_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
