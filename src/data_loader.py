import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os 

class PM25DataLoder:
    def __init__(self):
        self.path = 'data/raw/Almaty_PM2.5.csv'
    def load_raw_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found at {self.path}")
        df = pd.read_csv(self.path)
        return df 

    def preprocesses(self):
        df = self.load_raw_data()
        
        df['date'] = pd.to_datetime(df['period.datetimeTo.local'], errors='coerce', utc=True)
        
        def season_convert(x):
            if pd.isna(x):
                return 'unknown'
            if x in [1, 2, 12]:
                return 'winter'
            elif x in [3, 4, 5]:
                return 'spring'
            elif x in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        
        df['season'] = df['date'].dt.month.apply(season_convert)
        season_map = {'summer': 0, 'autumn': 1, 'spring': 2, 'winter': 3}
        df['season'] = df['season'].map(season_map)
        
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year
        
        drop_cols = [
            'coordinates', 'parameter.displayName', 'location_id', 'parameter.id',
            'parameter.name', 'parameter.units', 'period.label', 'period.interval',
            'period.datetimeFrom.local', 'period.datetimeTo.utc', 'period.datetimeTo.local',
            'period.datetimeFrom.utc', 'coverage.datetimeFrom.utc', 'coverage.datetimeFrom.local',
            'coverage.datetimeTo.utc', 'coverage.datetimeTo.local', 'coverage.expectedCount',
            'coverage.expectedInterval', 'coverage.observedInterval', 'coverage.observedCount',
            'coverage.percentComplete', 'summary.q02', 'summary.q98', 'summary.q25',
            'summary.median', 'summary.min', 'summary.max', 'summary.avg', 
            'summary.q75', 'summary.sd', 'date'
        ]
        df = df.drop(columns=drop_cols)
        
        df = df[df['value'] <= 300]
        df = df.dropna(subset=['value'])
        
        df['flagInfo.hasFlags'] = df['flagInfo.hasFlags'].astype(int)
        
        # lag features - previous days/weeks/years
        df = df.sort_values(['location_name', 'month', 'day_of_year'])
        
        df['pm25_lag1'] = df.groupby('location_name')['value'].shift(1)
        df['pm25_lag7'] = df.groupby('location_name')['value'].shift(7)
        df['pm25_rolling7'] = df.groupby('location_name')['value'].transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
        
        df = df.dropna(subset=['pm25_lag1', 'pm25_lag7', 'pm25_rolling7'])
        
        return df

    def train_test_split(self, test_size=0.2, random_state=42):
        df = self.preprocesses()
        y = df['value']
        X = df.drop(columns=['value'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=False
        )
        
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        train_enc_df = X_train.copy()
        train_enc_df['value'] = y_train.values
        train_enc = train_enc_df.groupby('location_name')['value'].mean()
        
        global_mean = y_train.mean()
        
        X_train['location_encoded'] = X_train['location_name'].map(train_enc).fillna(global_mean)
        X_test['location_encoded'] = X_test['location_name'].map(train_enc).fillna(global_mean)

        X_train = X_train.drop(columns=['location_name'])
        X_test = X_test.drop(columns=['location_name'])
        
        return X_train, X_test, y_train.values, y_test.values

if __name__ == '__main__':
    loader = PM25DataLoder()
    X_train, X_test, y_train, y_test = loader.train_test_split()
    df = loader.preprocesses()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data/processed")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "Cleaned_Almaty_PM2.5.csv"), index=False)
    print('data loaded successfully')




         
        
    