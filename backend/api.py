import base64
import io
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from sbcas_imputacao.benchmarking.experiment_runner import ExperimentRunner

app = FastAPI(title="Imputacao API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para desenvolvimento
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/describe")
async def describe_df(file: UploadFile = File(...)):

    """
    Shape e describe do dataframe passado
    """

    # if not file.filename.endswith(".parquet"):
    #     raise HTTPException(status_code=400, detail="Só .parquet são aceitos")

    if file.filename.endswith('.parquet'):
        df = pd.read_parquet(file.file)
    else:
        df = pd.read_csv(file.file)

    #try:
    #    df = pd.read_parquet(file.file)
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=f"Falha ao ler arquivo: {e}")

    runner = ExperimentRunner()
    result = runner.describe(df)

    return {
        "shape": {"rows": int(result["shape"][0]), "cols": int(result["shape"][1])},
        "describe": result["describe"].to_dict(),
    }

@app.post("/colunas_nans")
async def colunas_nans(file: UploadFile = File(...)):

    """
    Nome das colunas e NaNs em cada uma
    """

    if file.filename.endswith('.parquet'):
        df = pd.read_parquet(file.file)
    else:
        df = pd.read_csv(file.file)

    #try:
    #    df = pd.read_parquet(file.file)
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=f"falha ao ler df: {e}")

    runner = ExperimentRunner()
    result = runner.NaNs_each_column(df=df)
    return {col: int(count) for col, count in result.items()}


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


@app.post("/info_quadrantes")
async def info_quad(
    file: UploadFile = File(...),
    lista_features: List[str] = Query(...),
    window_size: int = Query(3, ge=1),
    test_window_size: int = Query(1, ge=1),
    n_split: int = Query(10, ge=1),
):
    """
    Informações dos quadrantes por feature
    """
    #try:
    #    df = pd.read_parquet(file.file)
    #except Exception as e:
    #    raise HTTPException(status_code=400, detail=f"falha ao ler df: {e}")

    if file.filename.endswith('.parquet'):
        df = pd.read_parquet(file.file)
    else:
        df = pd.read_csv(file.file)
    
    runner = ExperimentRunner(n_splits=n_split)
    results = runner.runners(
        df=df,
        lista_features=lista_features,
        window_size=window_size,
        test_window_size=test_window_size,
    )

    payload = []
    for item in results:
        fig = item["imagem"]
        payload.append({
            "feature": item["feature"],
            "erro_medio": item["erro_medio"].to_dict(),
            "imagem_base64": _fig_to_base64(fig),
        })
        plt.close(fig)

    return payload

@app.post("/imputar")
async def imputar_dados(
    arquivo: UploadFile = File(...),
    metodo: str = Form("mice"),
    features_a_imputar: str = Form("features_a_imputar"), 
    ignorar: str = Form(None)
):
    
    """
    Imputar dados faltantes das features selecionadas
    """

    caminho_entrada = f"temp_{arquivo.filename}"
    with open(caminho_entrada, "wb") as f:
        f.write(await arquivo.read())
        
    if caminho_entrada.endswith('.parquet'):
        df = pd.read_parquet(caminho_entrada)
    else:
        df = pd.read_csv(caminho_entrada)

    colunas_originais = df.columns.tolist()

    colunas_ignoradas = {}
    if ignorar:
        colunas_para_ignorar = [col.strip() for col in ignorar.split(',')]
        for col in colunas_para_ignorar:
            if col in df.columns:
                colunas_ignoradas[col] = df[col].copy()
                df = df.drop(columns=[col])

    if features_a_imputar:
        colunas_alvo = [col.strip() for col in features_a_imputar.split(',')]
    else:
        colunas_alvo = df.columns[df.isna().any()].tolist()

    runner = ExperimentRunner()
    df_imputed = df.copy()
    
    for col in colunas_alvo:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            df_imputed = runner.imputar(df=df_imputed, algoritmo=metodo, feature=col)

    for col, valores in colunas_ignoradas.items():
        df_imputed[col] = valores
        
    df_imputed = df_imputed[colunas_originais]

    caminho_saida = f"imputado_{arquivo.filename}"
    if caminho_saida.endswith('.parquet'):
        df_imputed.to_parquet(caminho_saida, index=False)
    else:
        df_imputed.to_csv(caminho_saida, index=False)
        
    os.remove(caminho_entrada)
    
    return FileResponse(path=caminho_saida, filename=caminho_saida)