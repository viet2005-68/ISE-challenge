import pandas as pd

def column_parser(path: str, sample_count: int):
    # path là đường dẫn file
    # sample_count là số sample rows muốn lấy ra
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
