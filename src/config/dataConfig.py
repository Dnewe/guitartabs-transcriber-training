from typing import Self, Dict, Any
from utils.fs_io import read_json


class DataConfig:
    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize with default values
            cls._instance._initialized = False
        return cls._instance
    
    def set_index_vars(self):
        Y_strings = []
        X_data = []
        for i,column in enumerate(self.DATA_HEADER):
            if ((str)(column)).startswith("y_"):
                Y_strings.append(i)
            elif ((str)(column)).startswith("x_"):
                X_data.append(i)
        setattr(self, "Y_STRINGS", Y_strings)
        setattr(self, "X_DATA", X_data)
    
    def initialize_from_json(self, json_path:str):
        if not self._initialized:
            metadata = read_json(json_path)
            for key, value in metadata.items():
                setattr(self, key.upper(), value)
            self._initialized = True
            self.set_index_vars()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            key.lower() : value
            for key, value in vars(self).items()
            if not key.startswith("_")
        }