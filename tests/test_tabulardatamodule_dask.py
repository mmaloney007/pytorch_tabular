#!/usr/bin/env python
# Author: Mike Maloney <mikey.maloney@gmail.com>
"""Tests for `pytorch_tabular` package with TabularDatamodule adjusted for Dask."""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# Import other necessary modules from pytorch_tabular
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the TabularDatamodule and TabularDataset from your local file
sys.path.insert(0, "/Users/maloney/Documents/GitHub/pytorch_tabular_soren/src/pytorch_tabular")
from tabular_datamodule import TabularDatamodule


# Fixtures for generating synthetic data
@pytest.fixture
def regression_data():
    """Fixture to generate synthetic regression data."""
    logger.info("Generating synthetic regression data...")
    np.random.seed(0)
    num_samples = 1000
    num_features = 6
    X = np.random.randn(num_samples, num_features)
    y = X @ np.random.randn(num_features) + np.random.randn(num_samples)
    df = pd.DataFrame(
        X,
        columns=[
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
    )
    df["MedHouseVal"] = y
    # Create a categorical column 'HouseAgeBin' with 5 categories
    df["HouseAgeBin"] = np.random.choice(["A", "B", "C", "D", "E"], size=num_samples)
    # Also, create a 'MedInc' column for multi-target regression
    df["MedInc"] = X[:, 0] * 0.5 + np.random.randn(num_samples)
    # Split into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    target = ["MedHouseVal"]
    logger.info("Regression data generated.")
    return train, test, target


@pytest.fixture
def timeseries_data():
    """Fixture to generate synthetic timeseries data."""
    logger.info("Generating synthetic timeseries data...")
    num_samples = 1000
    date_range = pd.date_range(start="1/1/2021", periods=num_samples, freq="H")
    data = pd.DataFrame(
        {
            "date": date_range,
            "Temperature": np.random.randn(num_samples),
            "Humidity": np.random.randn(num_samples),
            "Light": np.random.randn(num_samples),
            "CO2": np.random.randn(num_samples),
            "HumidityRatio": np.random.randn(num_samples),
            "Occupancy": np.random.randint(0, 2, size=num_samples),
        }
    )
    # Split into train and test
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    target = ["Occupancy"]
    logger.info("Timeseries data generated.")
    return train, test, target


@pytest.mark.parametrize("multi_target", [True, False])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        [],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None, "yeo-johnson"])
@pytest.mark.parametrize("normalize_continuous_features", [True, False])
@pytest.mark.parametrize(
    "target_transform",
    [
        None,
        PowerTransformer(method="yeo-johnson"),
        (lambda x: x**2, lambda x: np.sqrt(x)),
    ],
)
@pytest.mark.parametrize("validation_split", [None, 0.3])
@pytest.mark.parametrize("embedding_dims", [None, [(5, 1)]])
@pytest.mark.parametrize("cache_data", ["memory", "disk"])
def test_dataloader(
    regression_data,
    validation_split,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_transform,
    embedding_dims,
    cache_data,
    tmp_path_factory,
):
    logger.info("\nStarting test_dataloader...")
    (train, test, target) = regression_data
    train, valid = train_test_split(train, random_state=42)
    if len(continuous_cols) + len(categorical_cols) == 0:
        logger.info("No continuous or categorical columns. Skipping test.")
        return

    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
        validation_split=validation_split,
    )
    model_config_params = {"task": "regression", "embedding_dims": embedding_dims}
    model_config = CategoryEmbeddingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    config = tabular_model.config
    if cache_data and cache_data == "disk":
        cache_data = str(tmp_path_factory.mktemp("cache"))
    datamodule = TabularDatamodule(
        train=train,
        validation=valid,
        config=config,
        target_transform=target_transform,
        cache_data=cache_data,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    inferred_config = datamodule.update_config(config)
    if len(categorical_cols) > 0:
        logger.info(f"Checking categorical cardinality: {inferred_config.categorical_cardinality[0]}")
        assert inferred_config.categorical_cardinality[0] == 5
        if embedding_dims is None:
            logger.info(f"Checking inferred embedding dimensions: {inferred_config.embedding_dims[0][-1]}")
            assert inferred_config.embedding_dims[0][-1] == 3
        else:
            logger.info(f"Checking embedding dimensions: {inferred_config.embedding_dims[0][-1]}")
            assert inferred_config.embedding_dims[0][-1] == embedding_dims[0][-1]
    if normalize_continuous_features and len(continuous_cols) > 0 and cache_data not in [None, False]:
        mean = round(datamodule.train_dataset.data[config.continuous_cols[0]].mean())
        std = round(datamodule.train_dataset.data[config.continuous_cols[0]].std())
        logger.info(f"Mean of {config.continuous_cols[0]}: {mean}")
        logger.info(f"Std of {config.continuous_cols[0]}: {std}")
        assert mean == 0
        assert std == 1
    val_loader = datamodule.val_dataloader()
    _val_loader = datamodule.prepare_inference_dataloader(valid)
    chk_1 = next(iter(val_loader))["continuous"]
    chk_2 = next(iter(_val_loader))["continuous"]
    assert np.not_equal(chk_1, chk_2).sum().item() == 0
    logger.info("test_dataloader passed.")


@pytest.mark.parametrize(
    "freq",
    ["H", "D", "T", "S"],
)
def test_date_encoding(timeseries_data, freq):
    logger.info(f"\nStarting test_date_encoding with frequency: {freq}")
    (train, test, target) = timeseries_data
    train, valid = train_test_split(train, random_state=42)
    data_config = DataConfig(
        target=target + ["Occupancy"],
        continuous_cols=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
        categorical_cols=[],
        date_columns=[("date", freq, "%Y-%m-%d %H:%M:%S")],
        encode_date_columns=True,
    )
    model_config_params = {"task": "regression"}
    model_config = CategoryEmbeddingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    config = tabular_model.config
    datamodule = TabularDatamodule(
        train=train,
        validation=valid,
        config=config,
    )
    datamodule.prepare_data()
    if freq != "S":
        try:
            datamodule.setup("fit")
        except ValueError as e:
            logger.error(f"Error during datamodule setup: {e}")
            assert False, f"datamodule.setup('fit') failed with error: {e}"
        config = datamodule.config
        train_dataset = datamodule.train_dataset
        if freq == "H":
            expected_column = "date_Hour"
        elif freq == "D":
            expected_column = "date_Dayofyear"
        elif freq == "T":
            expected_column = "date_Minute"
        else:
            expected_column = None
        logger.info(f"Checking if expected date feature '{expected_column}' is present.")
        assert expected_column in train_dataset.data.columns
        logger.info("Date encoding test passed.")
    elif freq == "S":
        with pytest.raises(RuntimeError):
            datamodule.setup("fit")
        logger.info("Expected RuntimeError for frequency 'S' received.")


@pytest.mark.skipif(
    os.environ.get("SKIP_LARGE_TESTS") == "1", reason="Skipping large dataframe test due to resource constraints."
)
def test_large_dataframe():
    """Test with a DataFrame that exceeds memory to check Dask integration."""
    logger.info("\nStarting test_large_dataframe...")
    num_rows = 10_000_000  # 10 million rows
    num_features = 6
    import dask.dataframe as dd
    import dask.array as da

    logger.info("Generating large random data with Dask...")
    X = da.random.standard_normal(size=(num_rows, num_features), chunks=(1_000_000, num_features))
    y = da.random.standard_normal(size=(num_rows,), chunks=(1_000_000,))
    columns = [
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    df = dd.from_dask_array(X, columns=columns)
    df["MedHouseVal"] = y
    # Create a categorical column 'HouseAgeBin' with 5 categories
    df["HouseAgeBin"] = da.random.randint(0, 5, size=(num_rows,), chunks=1_000_000).map_blocks(
        lambda x: np.array(["A", "B", "C", "D", "E"])[x]
    )
    # Also, create a 'MedInc' column for multi-target regression
    df["MedInc"] = da.random.randn(num_rows, chunks=1_000_000)
    # No need to split since Dask can handle it
    train = df
    valid = None  # Let the datamodule handle validation split
    target = ["MedHouseVal"]

    data_config = DataConfig(
        target=target,
        continuous_cols=columns,
        categorical_cols=["HouseAgeBin"],
        continuous_feature_transform=None,
        normalize_continuous_features=True,
        validation_split=0.2,
    )
    model_config_params = {"task": "regression"}
    model_config = CategoryEmbeddingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    config = tabular_model.config

    # Instantiate the TabularDatamodule
    datamodule = TabularDatamodule(
        train=train,
        validation=valid,
        config=config,
        cache_data="memory",
    )

    # Prepare data and setup
    try:
        datamodule.prepare_data()
        datamodule.setup("fit")
        logger.info("Successfully set up datamodule with large dataframe.")
        # Now, get a batch from the dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Batch 'continuous' shape: {batch['continuous'].shape}")
        logger.info(f"Batch 'categorical' shape: {batch['categorical'].shape}")
        logger.info(f"Batch 'target' shape: {batch['target'].shape}")
        logger.info("test_large_dataframe passed.")
    except Exception as e:
        logger.error(f"Error during datamodule setup with large dataframe: {e}")
        assert False, f"test_large_dataframe failed with error: {e}"


# To run the tests, you can use the following command in your terminal:
# pytest test_tabular_databule_dask.py

# If you wish to skip the large dataframe test due to resource constraints, you can set the environment variable:
# export SKIP_LARGE_TESTS=1
# pytest test_tabular_databule_dask.py

# Verbose output
# pytest test_tabulardatamodule_dask.py -v


# Or modify the test_large_dataframe function to include the skipif decorator as shown.
