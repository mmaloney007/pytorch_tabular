# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Data Module adjusted to work with Dask for out-of-memory data."""

import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import dask.dataframe as dd
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dask_ml.preprocessing import (
    LabelEncoder as DaskLabelEncoder,
)
from dask_ml.preprocessing import (
    OrdinalEncoder as DaskOrdinalEncoder,
)
from dask_ml.preprocessing import (
    QuantileTransformer as DaskQuantileTransformer,
)
from dask_ml.preprocessing import (
    StandardScaler as DaskStandardScaler,
)
from omegaconf import DictConfig
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from torch.utils.data import DataLoader, Dataset

from pytorch_tabular.config import InferredConfig
from pytorch_tabular.utils import get_logger

logger = get_logger(__name__)


class DaskPowerTransformer(TransformerMixin):
    def __init__(self, method="yeo-johnson", standardize=True, copy=True, random_state=None):
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.random_state = random_state
        self._scikit_transformer = PowerTransformer(method=self.method, standardize=self.standardize, copy=self.copy)

    def fit(self, X, y=None):
        # Sample the data to fit the transformer
        X_sample = X.sample(frac=0.1, random_state=self.random_state).compute()
        self._scikit_transformer.fit(X_sample)
        return self

    def transform(self, X):
        return X.map_partitions(self._transform_partition)

    def inverse_transform(self, X):
        return X.map_partitions(self._inverse_transform_partition)

    def _transform_partition(self, df):
        transformed = self._scikit_transformer.transform(df)
        return pd.DataFrame(transformed, columns=df.columns, index=df.index)

    def _inverse_transform_partition(self, df):
        inversed = self._scikit_transformer.inverse_transform(df)
        return pd.DataFrame(inversed, columns=df.columns, index=df.index)


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        target: List[str] = None,
    ):
        """
        Dataset to load tabular data.

        Args:
            data (pd.DataFrame): Pandas DataFrame containing the data.
            task (str): Task type, either "classification" or "regression".
            continuous_cols (List[str], optional): List of continuous column names.
            categorical_cols (List[str], optional): List of categorical column names.
            target (List[str], optional): List of target column names.
        """
        self.task = task
        self.n = data.shape[0]
        self.target = target
        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        if task == "classification":
            self.y = self.y.astype(np.int64)
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values

        if self.categorical_cols:
            self.categorical_X = data[self.categorical_cols]
            self.categorical_X = self.categorical_X.astype(np.int64).values

    def __len__(self):
        """Returns the total number of samples."""
        return self.n

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return {
            "target": self.y[idx],
            "continuous": (self.continuous_X[idx] if self.continuous_cols else torch.Tensor()),
            "categorical": (self.categorical_X[idx] if self.categorical_cols else torch.Tensor()),
        }


class TabularDatamodule(pl.LightningDataModule):
    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": DaskQuantileTransformer,
            "params": {"output_distribution": "uniform", "random_state": None},
        },
        "quantile_normal": {
            "callable": DaskQuantileTransformer,
            "params": {"output_distribution": "normal", "random_state": None},
        },
        "box-cox": {
            "callable": DaskPowerTransformer,
            "params": {"method": "box-cox", "standardize": False, "random_state": None},
        },
        "yeo-johnson": {
            "callable": DaskPowerTransformer,
            "params": {"method": "yeo-johnson", "standardize": False},
        },
    }

    class CACHE_MODES(Enum):
        MEMORY = "memory"
        DISK = "disk"
        INFERENCE = "inference"

    def __init__(
        self,
        train: Union[pd.DataFrame, dd.DataFrame, str, Path],
        config: DictConfig,
        validation: Union[pd.DataFrame, dd.DataFrame, str, Path] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = 42,
        cache_data: str = "memory",
        copy_data: bool = True,
        verbose: bool = True,
    ):
        """
        The PyTorch Lightning Datamodule for tabular data.

        Args:
            train (Union[pd.DataFrame, dd.DataFrame, str, Path]): Training data.
            config (DictConfig): Configuration object.
            validation (Union[pd.DataFrame, dd.DataFrame, str, Path], optional): Validation data. Defaults to None.
            target_transform (Optional[Union[TransformerMixin, Tuple]], optional): Target transformation. Defaults to None.
            train_sampler (Optional[torch.utils.data.Sampler], optional): Sampler for training data. Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to 42.
            cache_data (str, optional): Cache mode ("memory" or "disk"). Defaults to "memory".
            copy_data (bool, optional): Whether to copy the data. Defaults to True.
            verbose (bool, optional): Verbosity flag. Defaults to True.
        """
        super().__init__()
        self.config = config
        self.seed = seed
        self.verbose = verbose
        self.train = self._load_data(train, copy_data)
        self.validation = self._load_data(validation, copy_data) if validation else None
        self._set_target_transform(target_transform)
        self.target = config.target or []
        self.batch_size = config.batch_size
        self.train_sampler = train_sampler
        self._fitted = False
        self._setup_cache(cache_data)
        self._inferred_config = self._update_config(config)

    @property
    def categorical_encoder(self):
        """Returns the categorical encoder."""
        return getattr(self, "_categorical_encoder", None)

    @categorical_encoder.setter
    def categorical_encoder(self, value):
        self._categorical_encoder = value

    @property
    def continuous_transform(self):
        """Returns the continuous transformer."""
        return getattr(self, "_continuous_transform", None)

    @continuous_transform.setter
    def continuous_transform(self, value):
        self._continuous_transform = value

    @property
    def scaler(self):
        """Returns the scaler for normalization."""
        return getattr(self, "_scaler", None)

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def label_encoder(self):
        """Returns the label encoder for target encoding."""
        return getattr(self, "_label_encoder", None)

    @label_encoder.setter
    def label_encoder(self, value):
        self._label_encoder = value

    @property
    def target_transforms(self):
        """Returns the target transformers."""
        if self.do_target_transform:
            return self._target_transforms
        else:
            return None

    @target_transforms.setter
    def target_transforms(self, value):
        self._target_transforms = value

    def _setup_cache(self, cache_data: Union[str, bool]) -> None:
        """Sets up the caching mechanism based on the cache_data parameter."""
        cache_data = cache_data.lower()
        if cache_data == self.CACHE_MODES.MEMORY.value:
            self.cache_mode = self.CACHE_MODES.MEMORY
        elif isinstance(cache_data, str):
            self.cache_mode = self.CACHE_MODES.DISK
            self.cache_dir = Path(cache_data)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(f"{cache_data} is not a valid path. Caching in memory")
            self.cache_mode = self.CACHE_MODES.MEMORY

    def _load_data(self, data, copy_data):
        """Loads data from a file path or DataFrame and converts it to a Dask DataFrame."""
        if isinstance(data, (str, Path)):
            data_path = Path(data)
            if data_path.suffix == ".csv":
                data = dd.read_csv(data)
            elif data_path.suffix in [".parquet", ".pq"]:
                data = dd.read_parquet(data)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")
        elif isinstance(data, pd.DataFrame):
            data = data.copy() if copy_data else data
            npartitions = max(1, len(data) // 1_000_000)
            data = dd.from_pandas(data, npartitions=npartitions)
        elif isinstance(data, dd.DataFrame):
            data = data.copy() if copy_data else data
        else:
            raise TypeError("Data must be a Pandas DataFrame, Dask DataFrame, or a file path.")
        return data

    def _set_target_transform(self, target_transform: Union[TransformerMixin, Tuple]) -> None:
        """Sets up the target transformation."""
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                target_transform = FunctionTransformer(func=target_transform[0], inverse_func=target_transform[1])
            self.do_target_transform = True
        else:
            self.do_target_transform = False
        self.target_transform_template = target_transform

    def _update_config(self, config) -> InferredConfig:
        """Calculates and updates key information in the config object."""
        categorical_dim = len(config.categorical_cols)
        continuous_dim = len(config.continuous_cols)
        if config.task == "regression":
            output_dim = len(config.target) if config.target else None
            output_cardinality = None
        elif config.task == "classification":
            if self.train is not None:
                output_cardinality = (
                    self.train[config.target].fillna("NA").nunique().compute().tolist() if config.target else None
                )
                output_dim = sum(output_cardinality)
            else:
                output_cardinality = (
                    self.train_dataset.data[config.target].fillna("NA").nunique().tolist() if config.target else None
                )
                output_dim = sum(output_cardinality)
        elif config.task == "ssl":
            output_cardinality = None
            output_dim = None
        else:
            raise ValueError(f"{config.task} is an unsupported task.")

        if self.train is not None:
            categorical_cardinality = (
                self.train[config.categorical_cols].fillna("NA").nunique().compute().astype(int).values.tolist()
            )
            categorical_cardinality = [x + 1 for x in categorical_cardinality]
        else:
            categorical_cardinality = (
                self.train_dataset.data[config.categorical_cols].nunique().compute().astype(int).values.tolist()
            )
            categorical_cardinality = [x + 1 for x in categorical_cardinality]

        if getattr(config, "embedding_dims", None) is not None:
            embedding_dims = config.embedding_dims
        else:
            embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_cardinality]

        return InferredConfig(
            categorical_dim=categorical_dim,
            continuous_dim=continuous_dim,
            output_dim=output_dim,
            output_cardinality=output_cardinality,
            categorical_cardinality=categorical_cardinality,
            embedding_dims=embedding_dims,
        )

    def update_config(self, config) -> InferredConfig:
        """Calculates and updates key information in the config object.

        Args:
            config (DictConfig): The configuration object.

        Returns:
            InferredConfig: The updated configuration object.
        """
        if self.cache_mode is self.CACHE_MODES.INFERENCE:
            warnings.warn("Cannot update config in inference mode. Returning the cached config")
            return self._inferred_config
        else:
            return self._update_config(config)

    def _encode_date_columns(self, data: dd.DataFrame) -> Tuple[dd.DataFrame, List[str]]:
        """Encodes date columns by extracting date features.

        Args:
            data (dd.DataFrame): The data containing date columns.

        Returns:
            Tuple[dd.DataFrame, List[str]]: The transformed data and list of added features.
        """
        added_features = []
        for field_name, freq, format in self.config.date_columns:
            data = self.make_date(data, field_name, format)
            data, new_feats = self.add_datepart(data, field_name, frequency=freq, prefix=None, drop=True)
            added_features += new_feats
        return data, added_features

    def _encode_categorical_columns(self, data: dd.DataFrame, stage: str) -> dd.DataFrame:
        """Encodes categorical columns using DaskOrdinalEncoder.

        Args:
            data (dd.DataFrame): The data containing categorical columns.
            stage (str): The stage, either "fit" or "inference".

        Returns:
            dd.DataFrame: The data with encoded categorical columns.
        """
        if stage != "fit":
            return self.categorical_encoder.transform(data)
        logger.debug("Encoding categorical columns using DaskOrdinalEncoder")
        self.categorical_encoder = DaskOrdinalEncoder(
            columns=self.config.categorical_cols,
            handle_unknown="use_encoded_value" if self.config.handle_unknown_categories else "error",
            unknown_value=-1,
            handle_missing="use_encoded_value" if self.config.handle_missing_values else "error",
            missing_value=-1,
        )
        data = self.categorical_encoder.fit_transform(data)
        return data

    def _transform_continuous_columns(self, data: dd.DataFrame, stage: str) -> dd.DataFrame:
        """Applies transformations to continuous columns.

        Args:
            data (dd.DataFrame): The data containing continuous columns.
            stage (str): The stage, either "fit" or "inference".

        Returns:
            dd.DataFrame: The data with transformed continuous columns.
        """
        transform = self.CONTINUOUS_TRANSFORMS[self.config.continuous_feature_transform]
        if "random_state" in transform["params"] and self.seed is not None:
            transform["params"]["random_state"] = self.seed
        if stage == "fit":
            self.continuous_transform = transform["callable"](**transform["params"])
            data[self.config.continuous_cols] = self.continuous_transform.fit_transform(
                data[self.config.continuous_cols]
            )
        else:
            data[self.config.continuous_cols] = self.continuous_transform.transform(data[self.config.continuous_cols])
        return data

    def _normalize_continuous_columns(self, data: dd.DataFrame, stage: str) -> dd.DataFrame:
        """Normalizes continuous columns.

        Args:
            data (dd.DataFrame): The data containing continuous columns.
            stage (str): The stage, either "fit" or "inference".

        Returns:
            dd.DataFrame: The data with normalized continuous columns.
        """
        if stage == "fit":
            self.scaler = DaskStandardScaler()
            data[self.config.continuous_cols] = self.scaler.fit_transform(data[self.config.continuous_cols])
        else:
            data[self.config.continuous_cols] = self.scaler.transform(data[self.config.continuous_cols])
        return data

    def _label_encode_target(self, data: dd.DataFrame, stage: str) -> dd.DataFrame:
        """Label encodes the target column(s) for classification tasks.

        Args:
            data (dd.DataFrame): The data containing target columns.
            stage (str): The stage, either "fit" or "inference".

        Returns:
            dd.DataFrame: The data with encoded target columns.
        """
        if self.config.task != "classification":
            return data
        if stage == "fit" or self.label_encoder is None:
            self.label_encoder = []
            for col in self.config.target:
                le = DaskLabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoder.append(le)
        else:
            for col, le in zip(self.config.target, self.label_encoder):
                data[col] = le.transform(data[col])
        return data

    def _target_transform(self, data: dd.DataFrame, stage: str) -> dd.DataFrame:
        """Applies transformations to the target column(s) for regression tasks.

        Args:
            data (dd.DataFrame): The data containing target columns.
            stage (str): The stage, either "fit" or "inference".

        Returns:
            dd.DataFrame: The data with transformed target columns.
        """
        if self.config.task != "regression" or not self.do_target_transform:
            return data
        if stage == "fit":
            self.target_transforms = []
            self.target_transformers = {}
            for col in self.config.target:
                transformer = copy.deepcopy(self.target_transform_template)
                transformer.fit(data[[col]])
                self.target_transformers[col] = transformer
            data[self.config.target] = data.map_partitions(
                lambda df: pd.DataFrame(
                    {col: self.target_transformers[col].transform(df[[col]]) for col in self.config.target}
                ),
                meta=data[self.config.target]._meta,
            )
        else:
            data[self.config.target] = data.map_partitions(
                lambda df: pd.DataFrame(
                    {col: self.target_transformers[col].transform(df[[col]]) for col in self.config.target}
                ),
                meta=data[self.config.target]._meta,
            )
        return data

    def preprocess_data(self, data: dd.DataFrame, stage: str = "inference") -> Tuple[dd.DataFrame, list]:
        """Preprocesses the data by encoding, transforming, and normalizing.

        Args:
            data (dd.DataFrame): The data to preprocess.
            stage (str, optional): The stage, either "fit" or "inference". Defaults to "inference".

        Returns:
            Tuple[dd.DataFrame, list]: The preprocessed data and list of added features.
        """
        added_features = None
        if self.config.encode_date_columns:
            data, added_features = self._encode_date_columns(data)
        if (added_features is not None) and (stage == "fit"):
            logger.debug(f"Added {added_features} features after encoding the date_columns")
            self.config.categorical_cols += added_features
            self.config.categorical_dim = (
                len(self.config.categorical_cols) if self.config.categorical_cols is not None else 0
            )
            self._inferred_config = self._update_config(self.config)
        if len(self.config.categorical_cols) > 0:
            data = self._encode_categorical_columns(data, stage)
        if (self.config.continuous_feature_transform is not None) and (len(self.config.continuous_cols) > 0):
            data = self._transform_continuous_columns(data, stage)
        if (self.config.normalize_continuous_features) and (len(self.config.continuous_cols) > 0):
            data = self._normalize_continuous_columns(data, stage)
        data = self._label_encode_target(data, stage)
        data = self._target_transform(data, stage)
        return data, added_features

    def _cache_dataset(self):
        """Caches the datasets either in memory or on disk based on the cache mode."""
        train_df = self.train.compute()
        val_df = self.validation.compute()
        train_dataset = TabularDataset(
            task=self.config.task,
            data=train_df,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=self.target,
        )
        self.train = None
        validation_dataset = TabularDataset(
            task=self.config.task,
            data=val_df,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=self.target,
        )
        self.validation = None

        if self.cache_mode is self.CACHE_MODES.DISK:
            torch.save(train_dataset, self.cache_dir / "train_dataset")
            torch.save(validation_dataset, self.cache_dir / "validation_dataset")
        elif self.cache_mode is self.CACHE_MODES.MEMORY:
            self.train_dataset = train_dataset
            self.validation_dataset = validation_dataset
        elif self.cache_mode is self.CACHE_MODES.INFERENCE:
            self.train_dataset = None
            self.validation_dataset = None
        else:
            raise ValueError(f"{self.cache_mode} is not a valid cache mode")

    def split_train_val(self, data: dd.DataFrame):
        """Splits the data into training and validation sets.

        Args:
            data (dd.DataFrame): The data to split.

        Returns:
            Tuple[dd.DataFrame, dd.DataFrame]: The training and validation data.
        """
        logger.debug(
            f"No validation data provided. Using {self.config.validation_split*100}% of train data as validation"
        )
        data = data.random_split(
            [1 - self.config.validation_split, self.config.validation_split],
            random_state=self.seed,
        )
        return data[0], data[1]

    def setup(self, stage: Optional[str] = None) -> None:
        """Performs data operations like splitting and preprocessing.

        Args:
            stage (Optional[str], optional): The stage, either "fit" or "inference". Defaults to None.
        """
        if not (stage is None or stage == "fit" or stage == "ssl_finetune"):
            return
        if self.verbose:
            logger.info(f"Setting up the datamodule for {self.config.task} task")
        is_ssl = stage == "ssl_finetune"
        if self.validation is None:
            self.train, self.validation = self.split_train_val(self.train)
        self.train, _ = self.preprocess_data(self.train, stage="fit" if not is_ssl else "inference")
        self.validation, _ = self.preprocess_data(self.validation, stage="inference")
        self._fitted = True
        self._cache_dataset()

    def inference_only_copy(self):
        """Creates a copy of the datamodule for inference-only scenarios.

        Returns:
            TabularDatamodule: A copy of the datamodule without training and validation datasets.
        """
        if not self._fitted:
            raise RuntimeError("Can create an inference-only copy only after the model is fitted.")
        dm_inference = copy.copy(self)
        dm_inference.train_dataset = None
        dm_inference.validation_dataset = None
        dm_inference.cache_mode = self.CACHE_MODES.INFERENCE
        return dm_inference

    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """Returns a list of time features appropriate for the given frequency string.

        Args:
            freq_str (str): Frequency string like "12H", "5min", "1D", etc.

        Returns:
            List[str]: List of time feature names.
        """
        features_by_offsets = {
            offsets.YearBegin: [],
            offsets.YearEnd: [],
            offsets.MonthBegin: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.MonthEnd: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.Week: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week",
            ],
            offsets.Day: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Weekday",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.BusinessDay: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Weekday",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.Hour: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Weekday",
                "Dayofweek",
                "Dayofyear",
                "Hour",
            ],
            offsets.Minute: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Weekday",
                "Dayofweek",
                "Dayofyear",
                "Hour",
                "Minute",
            ],
        }

        offset = to_offset(freq_str)

        for offset_type, features in features_by_offsets.items():
            if isinstance(offset, offset_type):
                return features

        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            Y, YS   - yearly
                alias: A
            M, MS   - monthly
            W       - weekly
            D       - daily
            B       - business days
            H       - hourly
            T       - minutely
                alias: min
        """
        raise RuntimeError(supported_freq_msg)

    @classmethod
    def make_date(cls, df: dd.DataFrame, date_field: str, date_format: str = None) -> dd.DataFrame:
        """Ensures that `df[date_field]` is of the right datetime type.

        Args:
            df (dd.DataFrame): The DataFrame containing the date field.
            date_field (str): The name of the date field.
            date_format (str, optional): The format of the date field. Defaults to None.

        Returns:
            dd.DataFrame: DataFrame with the date field converted to datetime.
        """
        field_dtype = df[date_field].dtype
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = dd.to_datetime(df[date_field], format=date_format)
        return df

    @classmethod
    def add_datepart(
        cls,
        df: dd.DataFrame,
        field_name: str,
        frequency: str,
        prefix: str = None,
        drop: bool = True,
    ) -> Tuple[dd.DataFrame, List[str]]:
        """Adds columns relevant to a date in the column `field_name` of `df`.

        Args:
            df (dd.DataFrame): DataFrame.
            field_name (str): Date field name.
            frequency (str): Frequency string like "12H", "5min", "1D", etc.
            prefix (str, optional): Prefix to add to the new columns. Defaults to None.
            drop (bool, optional): Whether to drop the original date field. Defaults to True.

        Returns:
            Tuple[dd.DataFrame, List[str]]: DataFrame with added columns and list of added columns.
        """
        field = df[field_name]
        prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
        attr = cls.time_features_from_frequency_str(frequency)
        added_features = []

        def add_features(df_part):
            field = df_part[field_name]
            for n in attr:
                if n == "Week":
                    df_part[prefix + "Week"] = field.dt.isocalendar().week
                else:
                    df_part[prefix + n] = getattr(field.dt, n.lower())
            if drop:
                df_part = df_part.drop(columns=[field_name])
            return df_part

        df = df.map_partitions(add_features)
        added_features = [prefix + n for n in attr]
        return df, added_features

    def _load_dataset_from_cache(self, tag: str = "train"):
        """Loads the dataset from cache based on the cache mode.

        Args:
            tag (str, optional): The dataset tag ("train" or "validation"). Defaults to "train".

        Returns:
            TabularDataset: The loaded dataset.
        """
        if self.cache_mode is self.CACHE_MODES.MEMORY:
            try:
                dataset = getattr(self, f"_{tag}_dataset")
            except AttributeError:
                raise AttributeError(f"{tag}_dataset not found in memory. Please provide the data for {tag} dataloader")
        elif self.cache_mode is self.CACHE_MODES.DISK:
            try:
                dataset = torch.load(self.cache_dir / f"{tag}_dataset")
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{tag}_dataset not found in {self.cache_dir}. Please provide the data for {tag} dataloader"
                )
        elif self.cache_mode is self.CACHE_MODES.INFERENCE:
            raise RuntimeError("Cannot load dataset in inference mode. Use `prepare_inference_dataloader` instead")
        else:
            raise ValueError(f"{self.cache_mode} is not a valid cache mode")
        return dataset

    @property
    def train_dataset(self) -> TabularDataset:
        """Returns the train dataset.

        Returns:
            TabularDataset: The train dataset.
        """
        return self._load_dataset_from_cache("train")

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def validation_dataset(self) -> TabularDataset:
        """Returns the validation dataset.

        Returns:
            TabularDataset: The validation dataset.
        """
        return self._load_dataset_from_cache("validation")

    @validation_dataset.setter
    def validation_dataset(self, value):
        self._validation_dataset = value

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Creates the training dataloader.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size or self.batch_size,
            shuffle=True if self.train_sampler is None else False,
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Creates the validation dataloader.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: The validation dataloader.
        """
        return DataLoader(
            self.validation_dataset,
            batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def _prepare_inference_data(self, df: dd.DataFrame) -> dd.DataFrame:
        """Prepares data for inference by applying necessary transformations.

        Args:
            df (dd.DataFrame): The data to prepare.

        Returns:
            dd.DataFrame: The prepared data.
        """
        if len(set(self.target) - set(df.columns)) > 0:
            if self.config.task == "classification":
                for i in range(len(self.target)):
                    df[self.config.target[i]] = -1
            else:
                df[self.config.target] = 0
        df, _ = self.preprocess_data(df, stage="inference")
        return df

    def prepare_inference_dataloader(
        self, df: Union[pd.DataFrame, dd.DataFrame], batch_size: Optional[int] = None, copy_df: bool = True
    ) -> DataLoader:
        """Prepares the dataloader for inference data.

        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The data for inference.
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
            copy_df (bool, optional): Whether to copy the DataFrame before processing. Defaults to True.

        Returns:
            DataLoader: The inference dataloader.
        """
        if copy_df:
            df = df.copy()
        if isinstance(df, pd.DataFrame):
            df = dd.from_pandas(df, npartitions=max(1, len(df) // 1_000_000))
        df = self._prepare_inference_data(df)
        df = df.compute()
        dataset = TabularDataset(
            task=self.config.task,
            data=df,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=(self.target if all(col in df.columns for col in self.target) else None),
        )
        return DataLoader(
            dataset,
            batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def save_dataloader(self, path: Union[str, Path]) -> None:
        """Saves the datamodule to a specified path.

        Args:
            path (Union[str, Path]): Path to save the datamodule.
        """
        if isinstance(path, str):
            path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load_datamodule(cls, path: Union[str, Path]):
        """Loads a datamodule from a specified path.

        Args:
            path (Union[str, Path]): Path to the datamodule.

        Returns:
            TabularDatamodule: The loaded datamodule.
        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        datamodule = joblib.load(path)
        return datamodule

    def copy(
        self,
        train: Union[pd.DataFrame, dd.DataFrame, str, Path],
        validation: Union[pd.DataFrame, dd.DataFrame, str, Path] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = None,
        cache_data: str = None,
        copy_data: bool = None,
        verbose: bool = None,
        call_setup: bool = True,
        config_override: Optional[Dict] = {},
    ):
        """Creates a copy of the datamodule with the option to override certain parameters.

        Args:
            train: Training data.
            validation: Validation data.
            target_transform: Target transformation.
            train_sampler: Sampler for training data.
            seed: Random seed.
            cache_data: Cache mode.
            copy_data: Whether to copy data.
            verbose: Verbosity flag.
            call_setup: Whether to call setup on the new datamodule.
            config_override: Dictionary to override config parameters.

        Returns:
            TabularDatamodule: The new datamodule copy.
        """
        if config_override is not None:
            for k, v in config_override.items():
                setattr(self.config, k, v)
        dm = TabularDatamodule(
            train=train,
            config=self.config,
            validation=validation,
            target_transform=target_transform or self.target_transforms,
            train_sampler=train_sampler or self.train_sampler,
            seed=seed or self.seed,
            cache_data=cache_data or self.cache_mode.value,
            copy_data=copy_data if copy_data is not None else True,
            verbose=verbose if verbose is not None else self.verbose,
        )
        dm.categorical_encoder = self.categorical_encoder
        dm.continuous_transform = self.continuous_transform
        dm.scaler = self.scaler
        dm.label_encoder = self.label_encoder
        dm.target_transforms = self.target_transforms
        if call_setup:
            dm.setup(stage="ssl_finetune")
        return dm
