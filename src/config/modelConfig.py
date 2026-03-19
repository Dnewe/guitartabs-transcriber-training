from typing import Self, Dict, Any, List
from utils.fs_io import read_json
import os


class ModelConfig:
    _instance = None
    LAYERS:Dict[str,List[str]]
    TRAIN_SIZE:int
    VAL_SIZE:int
    EPOCHS:int
    ALPHA:float
    BATCH_SIZE:int
    EARLYSTOP_PATIENCE:int
    MOMENTUM:float
    MEAN:float
    STD:float

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize with default values
            cls._instance._initialized = False
        return cls._instance
    
    def load_json(self, json_path:str):
        if not self._initialized:
            metadata = read_json(json_path)
            for key, value in metadata.items():
                setattr(self, key.upper(), value)
            self._initialized = True
    
    def load_default(self):
        self.load_json(os.path.join(os.curdir, "src", "config", "modelconfig.json"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            key.lower() : value
            for key, value in vars(self).items()
            if not key.startswith("_")
        }