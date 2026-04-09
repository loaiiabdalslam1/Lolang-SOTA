import logging
import os
import json
from datetime import datetime

class LolangLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"lolang_{datetime.now().strftime('%Y%m%d')}.log")
        self.json_log_file = os.path.join(log_dir, f"lolang_events_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("Lolang")

    def log_event(self, event_type: str, data: dict):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        with open(self.json_log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        self.logger.info(f"Event: {event_type} | Data: {data}")

# Global instance
logger = LolangLogger()
