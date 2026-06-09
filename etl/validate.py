"""Download the medical guidelines dataset for the RAG layer.

Source: epfl-llm/guidelines (HuggingFace)
Saves to: data/raw/medical_documents/guidelines_dataset
"""

import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "epfl-llm/guidelines"
OUTPUT_DIR = Path("data/raw/medical_documents/guidelines_dataset")


def download_guidelines(name: str = DATASET_NAME, output_dir: Path = OUTPUT_DIR) -> None:
    """Download the guidelines dataset and save it to disk."""
    try:
        logger.info("Downloading dataset: %s", name)
        dataset = load_dataset(name)

        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_dir))

        logger.info("Saved to %s", output_dir)
        logger.info("Splits: %s", list(dataset.keys()))
    except Exception as exc:
        logger.error("Failed to download %s: %s", name, exc)
        raise


if __name__ == "__main__":
    download_guidelines()