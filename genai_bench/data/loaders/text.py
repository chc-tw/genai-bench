from typing import Any, Dict, List, Set, Tuple, Union

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

from datasets import DatasetDict as HFDatasetDict

logger = init_logger(__name__)


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.

    TODO: Add support for prompt lambdas similar to ImageDatasetLoader.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.CSV,
        DatasetFormat.JSON,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Text"

    def _process_loaded_data(self, data: Any) -> List[Dict[str, str]]:
        """Process data loaded from dataset source."""
        # Handle data from dataset sources
        if isinstance(data, list):
            return [{"user_prompt": item} for item in data]

        # Handle dictionary data (from CSV files) or HuggingFace datasets
        prompt_column = self.dataset_config.prompt_column
        try:
            if isinstance(data, HFDatasetDict):
                available_splits = list(data.keys())
                if not available_splits:
                    raise ValueError(
                        "HuggingFace DatasetDict has no splits to select from."
                    )
                chosen_split = "train" if "train" in data else available_splits[0]
                data = data[chosen_split]
            column_data = data[prompt_column]
            # Ensure we return a list of strings
            if isinstance(column_data, list):
                return [{"user_prompt": str(item)} for item in column_data]
            else:
                # For HuggingFace datasets, convert to list
                if isinstance(column_data[0], list): # for sharegpt4o, column_data is a list of list of dicts
                    return [{"user_prompt": item[0]['value'], "assistant_prompt": item[1]['value']} for item in column_data]
                return [{"user_prompt": str(item)} for item in column_data]
        except (ValueError, KeyError) as e:
            # Provide helpful error message with available columns
            if isinstance(data, dict):
                available_columns = list(data.keys())
                raise ValueError(
                    f"Column '{prompt_column}' not found in CSV file. "
                    f"Available columns: {available_columns}"
                ) from e
            else:
                raise ValueError(
                    f"Cannot extract prompts from data: {type(data)}, error: {str(e)}"
                ) from e
