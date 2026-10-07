from src.config import (
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
)


def test_base_directory_exists():
    assert BASE_DIR.exists()


def test_data_directory_exists():
    assert DATA_DIR.exists()


def test_raw_data_directory_exists():
    assert RAW_DATA_DIR.exists()


def test_processed_data_directory_exists():
    assert PROCESSED_DATA_DIR.exists()


def test_model_directory_exists():
    assert MODEL_DIR.exists()