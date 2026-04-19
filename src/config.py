

# we use classes to store confuguration variables that are used across multiple files. 

import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
_config = yaml.safe_load((ROOT / "config.yaml").read_text())


class Paths:
    raw_data      = ROOT / _config["paths"]["raw_data"]
    cleaned_data  = ROOT / _config["paths"]["cleaned_data"]
    encoded_data  = ROOT / _config["paths"]["encoded_data"]
    model         = ROOT / _config["paths"]["model"]
    model_metadata= ROOT / _config["paths"]["model_metadata"]

class ModelConfig:
    duration_col  = _config["model"]["duration_col"]
    event_col     = _config["model"]["event_col"]
    cols_to_drop  = _config["model"]["cols_to_drop"]
