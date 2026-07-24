"""Data validation with Pandera (Owner: A). Requirement 6.

Checks schema, dtypes, nulls, value ranges, and target values before
training or inference.
"""
import sys
import pandera as pa
from pandera import Column, Check
import pandas as pd

# Kaggle/ULB creditcard.csv: Time, V1..V28 (PCA), Amount, Class
schema = pa.DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), coerce=True),
        **{f"V{i}": Column(float, nullable=False) for i in range(1, 29)},
        "Amount": Column(float, Check.ge(0)),
        "Class": Column(int, Check.isin([0, 1])),
    },
    strict=True,
)


def validate_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return schema.validate(df)  # raises SchemaError on failure


def main() -> None:
    path = sys.argv[sys.argv.index("--path") + 1] if "--path" in sys.argv else "data/raw/creditcard.csv"
    validate_file(path)
    print(f"OK: {path} passed validation")


if __name__ == "__main__":
    main()
