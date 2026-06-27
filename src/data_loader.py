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
        df['period.datetimeTo.local'] = pd.to_datetime(df['period.datetimeTo.local'], errors='coerce')
        def season_convert(x):
            if pd.isna(x):
                return 'unknown'
            if x in [1,2,12]:
                return 'winter'
            elif x in [3,4,5]:
                return 'spring'
            elif x in [6,7,8]:
                return 'summer'
            else:
                return 'autumn'
        df['season'] = df['period.datetimeTo.local'].dt.month.apply(season_convert)

        clean_df = df.drop(columns=['coordinates', 'parameter.displayName', 'location_id', 'parameter.id', 'parameter.name', 
        'parameter.units', 'parameter.displayName', 'parameter.displayName', 'period.label', 'period.interval', 
        'period.datetimeFrom.local', 'period.datetimeTo.utc', 'period.datetimeTo.local', 'period.datetimeFrom.utc', 
        'coverage.datetimeFrom.utc', 'coverage.datetimeFrom.local', 'coverage.datetimeTo.utc', 'coverage.datetimeTo.local',
        'coverage.expectedCount', 'coverage.expectedInterval', 'coverage.observedInterval', 'coverage.observedCount',
         'coverage.percentComplete', 'summary.q02', 'summary.q98', 'summary.q25', 'summary.median']) 
        clean_df = clean_df[clean_df['value'] <= 300]
        clean_df = clean_df.dropna()
        def season_label_encoder(x):
            if x == 'summer':
                return 0
            elif x == 'summer':
                return 1
            elif x == 'summer':
                return 2
            else:
                return 3
        clean_df['season'] = clean_df['season'].apply(season_label_encoder)
       
        le = LabelEncoder()
        clean_df['flagInfo.hasFlags'] = le.fit_transform(clean_df['flagInfo.hasFlags'])
        
        return clean_df 

    def train_test_split(self, test_size=0.2, random_state=42):
        df = self.preprocesses()
        y = df['value']
        X = df.drop(columns=['value'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
        return X_train, X_test, y_train.values, y_test.values

if __name__ == '__main__':
    loader = PM25DataLoder()
    X_train, X_test, y_train, y_test = loader.train_test_split()
    df = loader.preprocesses()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data/processed")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "Cleaned_Almaty_PM2.5.csv"), index=False)
    print('data loaded successfully')




         
        
    