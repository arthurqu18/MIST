import pandas as pd
import os
import torch
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from tabpfn import TabPFNRegressor
from huggingface_hub import login as hf_login

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
    
class tabpfn_imputer:
    def __init__(self, feature):
        # Autenticar com HuggingFace se o token estiver disponível
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)
            try:
                hf_login(token=hf_token, add_to_git_credential=False)
            except Exception as e:
                print(f"Aviso: Falha ao fazer login com token da HuggingFace: {e}")
        else:
            print("Aviso: Token da HuggingFace ausente. Defina HF_TOKEN ou HUGGINGFACE_HUB_TOKEN.")

        # TabPFN exige token para baixar pesos em ambientes não interativos
        tabpfn_token = os.getenv("TABPFN_TOKEN")
        if not tabpfn_token:
            raise RuntimeError(
                "TABPFN_TOKEN não encontrado no ambiente. "
                "Defina TABPFN_TOKEN (API Key do Prior Labs) antes de usar o método 'tabpfn'."
            )
        os.environ.setdefault("TABPFN_TOKEN", tabpfn_token)
        
        # Detectar se CUDA está disponível
        if torch.cuda.is_available():
            device = "cuda"
            print("TabPFN: CUDA detectado, usando GPU")
        else:
            device = "cpu"
            print("TabPFN: CUDA não disponível, usando CPU")
        
        # TabPFNRegressor com suporte a datasets grandes em CPU
        self._model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
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