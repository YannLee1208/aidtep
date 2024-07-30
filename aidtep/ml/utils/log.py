import loguru
from collections import defaultdict

class ProgressLogger:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.log_cache = ""

    def log_batch(self, prefix, batch_idx, batch_number, criterions: dict):
        log_info = f"{prefix}: [{batch_idx}/{batch_number}]"
        for name, value in criterions.items():
            log_info += f" {name}: {value:.4f} |"
        print(log_info, end='\r', flush=True)
        self.log_cache = log_info
        if batch_idx == batch_number:
            print()

    def finalize_log(self):
        loguru.logger.info(self.log_cache)
        self.log_cache = ""

    def reset(self):
        self.metrics = {}
        self.log_cache = ""

