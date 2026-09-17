from pythonjsonlogger import jsonlogger

import logging

class Formatter(jsonlogger.JsonFormatter):
    """
    Custom log formatter that emits log messages as JSON, with the "severity" field
    which Google Cloud uses to differentiate message levels.
    """
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = record.levelname

class Handler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.setFormatter(Formatter())
