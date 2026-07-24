"""Data validation with Pandera (Owner: A). Requirement 6.

Checks schema, dtypes, nulls, value ranges, and target values before
training or inference.
"""
import sys

import pandas as pd

# pandera >= 0.20 exposes the pandas API under pandera.pandas; older pins use
# the top-level module. Support both so the code matches requirements.txt
# (pinned) and a newer local install without a deprecation warning.
try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column
except ImportError:  # pragma: no cover - older pandera
    import pandera as pa
    from pandera import Check, Column

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
