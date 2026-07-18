import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from tabpfn_client import TabPFNRegressor, set_access_token


_TABPFN_AUTHENTICATED_TOKEN: str | None = None


def _get_tabpfn_api_key() -> str | None:
    return os.getenv("PRIORLABS_API_KEY") or os.getenv("TABPFN_TOKEN")


def _configure_tabpfn_client() -> None:
    global _TABPFN_AUTHENTICATED_TOKEN

    api_key = _get_tabpfn_api_key()
    if not api_key:
        raise RuntimeError(
            "Defina PRIORLABS_API_KEY (ou TABPFN_TOKEN) no ambiente para usar o TabPFN remoto."
        )

    if _TABPFN_AUTHENTICATED_TOKEN == api_key:
        return

    os.environ.setdefault("PRIORLABS_API_KEY", api_key)
    os.environ.setdefault("TABPFN_TOKEN", api_key)
    set_access_token(api_key)
    _TABPFN_AUTHENTICATED_TOKEN = api_key

class missforest:
    def __init__(self, max_iter: int = 20, random_state: int = 7, feature=None):
        self._model = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=7), 
                               max_iter=20, random_state=7)
        self.feature = feature
        self.columns_ = None
    
    def fit(self, df_train):
        self.columns_ = [column for column in df_train.columns if not df_train[column].isna().all()]
        self._model.fit(df_train[self.columns_])
    
    def transform(self, df_test) -> pd.DataFrame:
        df_imputed = df_test.copy()
        transformed = self._model.transform(df_test[self.columns_])
        df_imputed[self.columns_] = transformed
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)

class KNN:
    def __init__(self, n_neighbors: int = 10, feature=None):
        self.n_neighbors = n_neighbors
        self.feature = feature
        self._model = KNNImputer(n_neighbors=n_neighbors)
        self.columns_ = None
    
    def fit(self, df_train):
        self.columns_ = [column for column in df_train.columns if not df_train[column].isna().all()]
        self._model.fit(df_train[self.columns_])

    def transform(self, df_test) -> pd.DataFrame:
        df_imputed = df_test.copy()
        transformed = self._model.transform(df_test[self.columns_])
        df_imputed[self.columns_] = transformed
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)

class MICE:
    def __init__(self, max_iter: int = 20, random_state: int = 7, feature=None):
        self._model = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=7)
        self.feature = feature
        self.columns_ = None

    def fit(self, df_train):
        self.columns_ = [column for column in df_train.columns if not df_train[column].isna().all()]
        self._model.fit(df_train[self.columns_])

    def transform(self, df_test) -> pd.DataFrame:
        df_imputed = df_test.copy()
        transformed = self._model.transform(df_test[self.columns_])
        df_imputed[self.columns_] = transformed
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)
    
class Mean:
    @staticmethod
    def fit_transform(df_train, df_test, feature) -> pd.DataFrame:
        df_imputed = df_test.copy()
        mean = df_train[feature].mean()
        df_imputed[feature] = df_test[feature].fillna(mean)
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)

class Zero:
    @staticmethod
    def fit_transform(df_train, df_test, feature) -> pd.DataFrame:
        df_imputed = df_test.copy()
        df_imputed[feature] = df_test[feature].fillna(value=0)
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)
    
class tabpfn_imputer:
    def __init__(self, feature):
        _configure_tabpfn_client()
        self._model = TabPFNRegressor(ignore_pretraining_limits=True)
        self.feature = feature
        self.x_columns_ = None
    
    def fit(self, df_train):
        X_train = df_train.drop(columns=[self.feature])
        self.x_columns_ = [column for column in X_train.columns if not X_train[column].isna().all()]
        y_train = df_train[self.feature]
        self._model.fit(X_train[self.x_columns_].values, y_train.values)
    
    def transform(self, df_test) -> pd.DataFrame:
        df_imputed = df_test.copy()    
        preds = self._model.predict(df_test.drop(columns=[self.feature])[self.x_columns_].values)
        df_imputed[self.feature] = preds
        return pd.DataFrame(df_imputed, columns=df_test.columns, index=df_test.index)