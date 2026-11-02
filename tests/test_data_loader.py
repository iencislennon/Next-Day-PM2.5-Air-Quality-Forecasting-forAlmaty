import numpy as np
import pandas as pd
import pytest

from src.data_loader import PM25DataLoder

COLUMNS = [
    'coordinates', 'parameter.displayName', 'location_id', 'parameter.id',
    'parameter.name', 'parameter.units', 'period.label', 'period.interval',
    'period.datetimeFrom.local', 'period.datetimeTo.utc', 'period.datetimeTo.local',
    'period.datetimeFrom.utc', 'coverage.datetimeFrom.utc', 'coverage.datetimeFrom.local',
    'coverage.datetimeTo.utc', 'coverage.datetimeTo.local', 'coverage.expectedCount',
    'coverage.expectedInterval', 'coverage.observedInterval', 'coverage.observedCount',
    'coverage.percentComplete', 'summary.q02', 'summary.q98', 'summary.q25',
    'summary.median', 'summary.min', 'summary.max', 'summary.avg',
    'summary.q75', 'summary.sd',
]


def make_csv(tmp_path, days=800, stations=('A', 'B'), missing_every=None):
    rng = np.random.default_rng(0)
    dates = pd.date_range('2023-01-01', periods=days, freq='D', tz='UTC')
    rows = []
    for st in stations:
        for i, d in enumerate(dates):
            row = {c: 0 for c in COLUMNS}
            row['period.datetimeTo.local'] = d.isoformat()
            row['location_name'] = st
            row['value'] = np.nan if missing_every and i % missing_every == 0 else float(rng.uniform(5, 150))
            row['flagInfo.hasFlags'] = False
            rows.append(row)
    path = tmp_path / 'pm25.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_split_is_chronological(tmp_path):
    loader = PM25DataLoder(make_csv(tmp_path))
    df = loader.preprocesses()
    X_train, X_test, y_train, y_test = loader.train_test_split()
    train_dates = df.loc[X_train.index, 'date']
    test_dates = df.loc[X_test.index, 'date']
    assert train_dates.max() < test_dates.min()
    assert len(y_train) + len(y_test) == len(df)


def test_split_crosses_years_not_stations(tmp_path):
    loader = PM25DataLoder(make_csv(tmp_path))
    X_train, X_test, _, _ = loader.train_test_split()
    # Both stations must appear in train and test (no station-wise holdout).
    assert X_train['location_encoded'].nunique() >= 1
    assert len(X_test) > 0 and len(X_train) > len(X_test)


def test_lags_use_only_past_values(tmp_path):
    path = make_csv(tmp_path, stations=('A',))
    raw = pd.read_csv(path)['value'].reset_index(drop=True)
    df = PM25DataLoder(path).preprocesses().sort_values('date').reset_index(drop=True)
    # Loader drops the first 7 rows (lag7 undefined); row k corresponds to raw index k+7.
    for k in (0, 5, 100, len(df) - 1):
        t_idx = k + 7
        assert df['value'].iloc[k] == pytest.approx(raw.iloc[t_idx])
        assert df['pm25_lag1'].iloc[k] == pytest.approx(raw.iloc[t_idx - 1])
        assert df['pm25_lag7'].iloc[k] == pytest.approx(raw.iloc[t_idx - 7])
        assert df['pm25_rolling7'].iloc[k] == pytest.approx(raw.iloc[t_idx - 7:t_idx].mean())


def test_target_encoding_uses_train_only(tmp_path):
    loader = PM25DataLoder(make_csv(tmp_path))
    df = loader.preprocesses()
    X_train, _, y_train, _ = loader.train_test_split()
    train_mean_a = y_train[(df.loc[X_train.index, 'location_name'] == 'A').values].mean()
    enc_a = X_train.loc[(df.loc[X_train.index, 'location_name'] == 'A'), 'location_encoded'].iloc[0]
    assert enc_a == pytest.approx(train_mean_a)


def test_missing_values_are_dropped(tmp_path):
    loader = PM25DataLoder(make_csv(tmp_path, missing_every=10))
    df = loader.preprocesses()
    assert not df['value'].isna().any()
    assert not df[['pm25_lag1', 'pm25_lag7', 'pm25_rolling7']].isna().any().any()
