import pandas as pd
import pandera as pa
import pytest
from src.validate import schema


def _valid_row():
    row = {"Time": 0.0, "Amount": 10.0, "Class": 0}
    row.update({f"V{i}": 0.1 for i in range(1, 29)})
    return pd.DataFrame([row])


def test_valid_row_passes():
    schema.validate(_valid_row())


def test_bad_target_fails():
    df = _valid_row().assign(Class=2)
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)


def test_negative_amount_fails():
    df = _valid_row().assign(Amount=-1.0)
    with pytest.raises(pa.errors.SchemaError):
        schema.validate(df)
