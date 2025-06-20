import pandas as pd
from typing import Dict, Any

def column_parser(path: str, sample_count: int) -> Dict[str, Any]:
    # path là đường dẫn file
    # sample_count là số sample rows muốn lấy ra

    """
    Parse the columns and some sample rows of a csv file.

    Parameters:
        path (str): The path to the target csv file.
        sample_count (int): The number of sample rows to return.

    Returns:
        A dict of the columns and some sample rows parsed from the target csv file.
    """
    if not path.endswith(".csv"):
        raise ValueError("Provided file is not a CSV.")

    try:
        df = pd.read_csv(path)
        columns = df.columns.to_list()
        sample_rows = df.head(sample_count)

        return {
            "columns": columns,
            "sample_rows": sample_rows
        }
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: '{path}'")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse the CSV file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while reading CSV: {e}")
