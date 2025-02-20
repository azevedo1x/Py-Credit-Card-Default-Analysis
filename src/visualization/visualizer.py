import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
from pathlib import Path

class Visualizer:
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.style_config = {
            'figsize': (12, 8),
            'palette': 'viridis',
            'font_scale': 1.2
        }
        self._set_style()
        
    