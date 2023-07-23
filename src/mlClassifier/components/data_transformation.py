from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from mlClassifier.entity import DataTransformationConfig
from mlClassifier.utils.common import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save(self):
        df = pd.read_csv(self.config.data_path)
        df = df.drop_duplicates()

        X, y = df.drop(columns=['diabetes'],axis=1), pd.DataFrame(df['diabetes'])

        numerical_columns = list(X.select_dtypes(exclude="object").columns)
        categorical_columns = list(X.select_dtypes(include="object").columns)

        num_pipeline = Pipeline (
            steps=[
                ("scaler", MinMaxScaler())        
            ]
        )
        cat_pipeline = Pipeline (
            steps=[
                ("one_hot_encoder", OneHotEncoder()),
            ]
        )

        preprocessor = ColumnTransformer (
            [
                ("num_pipline", num_pipeline, numerical_columns),
                ("cat_pipline", cat_pipeline, categorical_columns)
            ]
        )

        X = pd.DataFrame(preprocessor.fit_transform(X))
        y.reset_index(drop=True, inplace=True)
        df = pd.concat([X, y], axis=1)
        df.to_csv(self.config.save_path, index=False)

        save_object(file_path=self.config.preprocessor_path, obj=preprocessor)