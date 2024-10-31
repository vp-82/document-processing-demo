from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import json
from enum import Enum


class DeviceType(str, Enum):
    """Supported device types."""
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


@dataclass
class ModelConfig:
    """Configuration for the LayoutLM model."""
    model_name: str = "microsoft/layoutlmv3-base"
    max_seq_length: int = 512
    image_size: tuple[int, int] = (224, 224)
    apply_ocr: bool = True


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 2
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2


@dataclass
class DataConfig:
    """Configuration for data handling."""
    dataset_dir: Path
    output_dir: Path
    train_dir_name: str = "train"
    validation_dir_name: str = "validation"
    supported_formats: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    validation_split: float = 0.2
    num_workers: int = 0  # Set to 0 for M1 Mac compatibility


class Config:
    """Main configuration class."""
    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Initialize configurations
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig(
            dataset_dir=self.dataset_dir,
            output_dir=self.output_dir
        )
        
        # Set device
        self.device = self._get_device()
        
        # Validate configuration
        self._validate_config()

    def _get_device(self) -> torch.device:
        """Determine the appropriate device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Validate directories
        if not self.data.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: \
                             {self.data.dataset_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate training parameters
        if self.training.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if self.training.num_epochs < 1:
            raise ValueError("Number of epochs must be positive")
        if not 0 <= self.data.validation_split <= 1:
            raise ValueError("Validation split must be between 0 and 1")

    def save(self, path: Optional[str | Path] = None) -> None:
        """Save configuration to JSON file."""
        if path is None:
            path = self.output_dir / "config.json"
        else:
            path = Path(path)

        config_dict = {
            "model": vars(self.model),
            "training": vars(self.training),
            "data": {
                "dataset_dir": str(self.data.dataset_dir),
                "output_dir": str(self.data.output_dir),
                "train_dir_name": self.data.train_dir_name,
                "validation_dir_name": self.data.validation_dir_name,
                "supported_formats": list(self.data.supported_formats),
                "validation_split": self.data.validation_split,
                "num_workers": self.data.num_workers
            },
            "device": str(self.device)
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        data_config = DataConfig(
            dataset_dir=Path(config_dict["data"]["dataset_dir"]),
            output_dir=Path(config_dict["data"]["output_dir"]),
            train_dir_name=config_dict["data"]["train_dir_name"],
            validation_dir_name=config_dict["data"]["validation_dir_name"],
            supported_formats=tuple(config_dict["data"]["supported_formats"]),
            validation_split=config_dict["data"]["validation_split"],
            num_workers=config_dict["data"]["num_workers"]
        )

        return cls(
            dataset_dir=data_config.dataset_dir,
            output_dir=data_config.output_dir,
            model_config=model_config,
            training_config=training_config,
            data_config=data_config
        )


def get_default_config(dataset_dir: str | Path, 
                       output_dir: str | Path) -> Config:
    """Create a default configuration."""
    return Config(
        dataset_dir=dataset_dir,
        output_dir=output_dir
    )
