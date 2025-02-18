import pandas as pd
import logging
from typing import Optional, Tuple, Union
from pathlib import Path

class DataLoader:
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else None
        self.data = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data: Optional[Union[list, dict]] = None) -> pd.DataFrame:
        try:
            if data:
                self.data = pd.DataFrame(data)
            elif self.data_path:
                file_readers = {
                    ".xls": pd.read_excel,
                    ".xlsx": pd.read_excel,
                    ".csv": pd.read_csv,
                }
                
                extension = self.data_path.suffix
                if extension in file_readers:
                    self.data = file_readers[extension](self.data_path, header=0)
                else:
                    raise ValueError(f"Unsupported file format: {extension}")
            else:
                raise ValueError("No data source provided")

            self._validate_data()
            return self.data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

   
   