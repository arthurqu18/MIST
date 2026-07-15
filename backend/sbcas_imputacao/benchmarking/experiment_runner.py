from pygrinder import mcar
from sklearn.model_selection import KFold
from ..imputadores import tabpfn_imputer, KNN, Mean, missforest, MICE
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# mostrar todas as colunas
pd.set_option('display.max_columns', None)
# não quebrar quadro em múltiplas linhas (force uma linha por índice)
pd.set_option('display.expand_frame_repr', False)
# largura máxima em caracteres — aumente se necessário
pd.set_option('display.width', 1200)
# formato de float (opcional)
pd.set_option('display.float_format', '{:.2f}'.format)

class ExperimentRunner:
    def __init__(self, n_splits: int = 10, random_state: int = 78):
        self.n_splits = n_splits
        self.random_state = random_state

    def calculate_errors(self, df_true, df_imputed, column):
        df_subset = df_true.loc[df_imputed.index, :]
        mae = mean_absolute_error(df_subset[column], df_imputed[column])
        mse = mean_squared_error(df_subset[column], df_imputed[column])
        return mae, mse

    def run(self, df: pd.DataFrame, feature, window_size: int = 3, test_window_size: int = 1):      
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' não encontrada no dataframe passado")
        if window_size < 1:
            raise ValueError("window_size precisa ser maior ou igual a 1")
        if test_window_size < 1:
            raise ValueError("test_window_size precisa ser maior ou igual a 1")
        
        self.feature = feature

        # mcar
        df_copy = df.copy()
        df_copy = df_copy.dropna(subset=[self.feature])
        df_mcar = df_copy.copy()
        df_mcar[self.feature] = mcar(df_copy[[self.feature]].values, p=0.50)

        # kfold
        idx_missing = df_mcar[df_mcar[self.feature].isna()].index
        df_nans = df_mcar.loc[idx_missing]
        df_controle = df.loc[idx_missing]

        
        imputadores = {'média': Mean,'knn': KNN,'mice': MICE, 'tabpfn': tabpfn_imputer, 'missforest': missforest}
        dfs = {nome: [] for nome in imputadores.keys()}
        metricas = []

        if self.n_splits < window_size + test_window_size:
            raise ValueError(
                "n_splits precisa ser maior ou igual à soma de window_size e test_window_size"
            )

        # Cada fold é um bloco ordenado de linhas; a janela anda sobre esses blocos.
        fold_indices = [test_idx for _, test_idx in KFold(n_splits=self.n_splits, shuffle=False).split(df_nans)]
        rolling_steps = range(0, len(fold_indices) - window_size - test_window_size + 1, test_window_size)

        for window_start in rolling_steps:
            train_fold_indices = fold_indices[window_start: window_start + window_size]
            test_fold_indices = fold_indices[
                window_start + window_size: window_start + window_size + test_window_size
            ]

            train_idx = np.concatenate(train_fold_indices)
            test_idx = np.concatenate(test_fold_indices)

            df_train = df_controle.iloc[train_idx]
            df_test = df_nans.iloc[test_idx]

            for nome, func in imputadores.items():
                start_time = time.time()

                if(nome != "média"):
                    imputer = func(feature=self.feature)
                    imputer.fit(df_train.copy())
                    df_imputed = imputer.transform(df_test.copy())
                else:
                    df_imputed = func.fit_transform(df_train=df_train.copy(), df_test=df_test.copy(), feature=self.feature)

                tempo = time.time() - start_time
                mae, mse = self.calculate_errors(df_controle, df_imputed, self.feature)
                metricas.append({
                    'metodo': nome,
                    'mae': mae,
                    'mse': mse,
                    'tempo_segundos': tempo
                })

                # logica quadrantes
                dfs[nome].append(df_imputed[[self.feature]])

        
        df_metricas = pd.DataFrame(metricas)
        print(df_metricas.groupby('metodo')[['mae', 'mse', 'tempo_segundos']].mean())

        for nome in dfs.keys():
            dfs[nome] = pd.concat(dfs[nome], axis=0)
        
        results_final = pd.concat(dfs, axis=1)
        results_final.columns = results_final.columns.droplevel(1)

        df_controle_copy = df_controle.copy()
        imputador = KNNImputer(n_neighbors=10)
        colunas_validas = [column for column in df_controle_copy.columns if not df_controle_copy[column].isna().all()]
        df_notNan = df_controle_copy.copy()
        df_notNan[colunas_validas] = imputador.fit_transform(df_controle_copy[colunas_validas])

        # normalização mantendo DataFrame
        scaler = MinMaxScaler()
        df_select = pd.DataFrame(
            scaler.fit_transform(df_notNan),
            columns=df_controle_copy.columns,
            index=df_controle_copy.index
        )

        nbrs = NearestNeighbors(n_neighbors=11, metric="euclidean") 
        nbrs.fit(df_select)  

        distances, indices = nbrs.kneighbors(df_select)  

        distancia_media_dict = {
            index: distances[i, 1:].mean()
            for i, index in enumerate(df_select.index)
            if index in results_final.index
        }
        results_final["Distancia Media"] = results_final.index.map(distancia_media_dict)

        entropia_media_dict = {}
        for i, index in enumerate(df_select.index):
            if index in results_final.index:
                vizinhos_indices = indices[i, 1:] 

                vizinhos_valores = df_select.iloc[vizinhos_indices].values.flatten()

                kde = gaussian_kde(vizinhos_valores)
                pdf_values = kde(vizinhos_valores)

                entropy = -np.sum(pdf_values * np.log(pdf_values + 1e-10)) / len(vizinhos_valores)

                entropia_media_dict[index] = entropy

        results_final["Entropia"] = results_final.index.map(entropia_media_dict)

        for metodo in imputadores.keys():
            results_final[f"Erro Absoluto {metodo}"] = abs(results_final[metodo] - df_controle[self.feature])

        media_x = results_final["Distancia Media"].mean()
        media_y = results_final["Entropia"].mean()

        results_final["quadrante"] = np.where(results_final["Distancia Media"] >= media_x, "Direita", "Esquerda")
        results_final["quadrante"] += np.where(results_final["Entropia"] >= media_y, " Superior", " Inferior")

        erro_medio = results_final.groupby("quadrante")[["Erro Absoluto média", "Erro Absoluto knn","Erro Absoluto mice","Erro Absoluto missforest", "Erro Absoluto tabpfn"]].mean()
        erro_medio = erro_medio.round(2)
        print(erro_medio)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=results_final,
            x="Distancia Media",
            y="Entropia",
            alpha=0.7,
            color="lightseagreen",
            edgecolor="black",
            s=30,
            ax=ax,
        )
        ax.set_title(f"Distribuição para a feature {feature}")
        ax.axvline(media_x, color="red", linestyle="--", linewidth=1.5, label=f"X = {media_x:.2f}")
        ax.axhline(media_y, color="blue", linestyle="--", linewidth=1.5, label=f"Y = {media_y:.2f}")

        quadrantes = ["Esquerda Inferior", "Esquerda Superior", "Direita Inferior", "Direita Superior"]
        metrics_text = f"Erro por Quadrante:\n"

        for quad in quadrantes:
            metrics_text += f"• {quad}: {erro_medio['Erro Absoluto knn'].get(quad, 0):.2f}\n"
        
        ax.text(
            1.05, 0.95,               # Coordinates: slightly right of the plot (x=1.05), near the top (y=0.95)
            metrics_text.strip(), 
            transform=ax.transAxes,   # Tells matplotlib to use relative sizing instead of data values
            fontsize=11, 
            verticalalignment='top',  # Aligns the top of the text block to your y coordinate
            fontfamily='monospace',   # Monospace keeps numbers perfectly aligned vertically
            bbox=dict(                # Adds a beautiful background box around your text block
            boxstyle="round,pad=0.5", 
            facecolor="#f9f9f9", 
            edgecolor="gray", 
            alpha=0.9
            )
        )        
                

        return {
            "erro_medio": erro_medio,
            "imagem": fig,
        }
            
    # executa `run` varias vezes --> para gerar imagens quadrantes
    def runners(self, df: pd.DataFrame, lista_features, window_size: int = 3, test_window_size: int = 1):
        results = []
        for feature in lista_features:
            result = self.run(
                df=df,
                feature=feature,
                window_size=window_size,
                test_window_size=test_window_size,
            )
            results.append({
                "feature": feature,
                "erro_medio": result["erro_medio"],
                "imagem": result["imagem"],
            })
        return results
        
    # imputação de dataframe
    def imputar(self, df: pd.DataFrame, algoritmo, feature):
        df_copy = df.copy()
        print("Nans antes: ", df[feature].isna().sum())
        df_train = df_copy[~(df_copy[feature].isna())]
        df_test = df_copy[(df_copy[feature].isna())]

        match algoritmo:
            case "média": 
                df_imputed = Mean.fit_transform(df_train=df_train.copy(), df_test=df_test.copy(), feature=feature)
            case "knn":
                imputer = KNN(feature=feature)
                imputer.fit(df_train.copy())
                df_imputed = imputer.transform(df_test.copy())
            case "mice":
                imputer = MICE(feature=feature)
                imputer.fit(df_train.copy())
                df_imputed = imputer.transform(df_test.copy())
            case "tabpfn":
                imputer = tabpfn_imputer(feature=feature)
                imputer.fit(df_train.copy())
                df_imputed = imputer.transform(df_test.copy())
            case "missforest":
                imputer = missforest(feature=feature)
                imputer.fit(df_train.copy())
                df_imputed = imputer.transform(df_test.copy())

        df_copy.loc[df_imputed.index, feature] = df_imputed[feature]
        print("NaNs depois: ", df_copy[feature].isna().sum())
        return pd.DataFrame(df_copy)
    
    # shape do df e describe 
    def describe(self, df: pd.DataFrame):
        return {"shape": df.shape, "describe": df.describe()}
    
    # nome das colunas
    def features_names(self, df: pd.DataFrame):
        return df.columns
    
    # NaNs por feature
    def NaNs_each_column(self, df: pd.DataFrame):
        dic = {}
        for column in df.columns:
            dic[column] = df[column].isna().sum()
        return dic