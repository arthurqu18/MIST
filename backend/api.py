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


def _is_tabpfn_auth_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "invalid token" in message or "401" in message


def _raise_tabpfn_auth_error(exc: RuntimeError) -> None:
    raise HTTPException(
        status_code=400,
        detail="A API key da Prior Labs é inválida para o TabPFN ou está desatualizada.",
    ) from exc


@app.post("/describe")
async def describe_df(file: UploadFile = File(...)):

    """
    Shape e describe do dataframe passado
    """

    if file.filename.endswith('.parquet'):
        df = pd.read_parquet(file.file)
    else:
        df = pd.read_csv(file.file)

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
    lista_algoritmos: List[str] | None = Query(None),
    window_size: int = Query(3, ge=1),
    test_window_size: int = Query(1, ge=1),
    n_splits: int = Query(10, ge=1),
    usar_tabpfn: bool = Query(True),
    priorlabs_api_key: str | None = Query(None),
):
    
    if file.filename.endswith('.parquet'):
        df = pd.read_parquet(file.file)
    else:
        df = pd.read_csv(file.file)

    if n_splits < window_size + test_window_size:
            raise HTTPException(
                status_code=400,
                detail="A quantidade de folds (n_splits) precisa ser maior ou igual à soma de window_size e test_window_size.",
            )

    if lista_algoritmos is not None and len(lista_algoritmos) == 0:
        raise HTTPException(
            status_code=400,
            detail="Selecione ao menos um algoritmo para avaliação.",
        )

    tabpfn_solicitado = (
        "tabpfn" in lista_algoritmos if lista_algoritmos is not None else usar_tabpfn
    )

    if tabpfn_solicitado and not usar_tabpfn:
        raise HTTPException(
            status_code=400,
            detail="TabPFN foi desativado na configuração acima. Ative 'Usar TabPFN' ou desmarque o TabPFN nos algoritmos.",
        )

    if tabpfn_solicitado and not priorlabs_api_key:
        raise HTTPException(
            status_code=400,
            detail="Não é possível rodar o TabPFN sem a chave API. Favor desmarcar o TabPFN ou inserir a API Key.",
        )
    
    runner = ExperimentRunner(n_splits=n_splits)

    try:
        results = runner.runners(
            df=df,
            lista_features=lista_features,
            window_size=window_size,
            test_window_size=test_window_size,
            usar_tabpfn=usar_tabpfn,
            tabpfn_api_key=priorlabs_api_key,
            lista_algoritmos=lista_algoritmos,
        )
    except RuntimeError as exc:
        if usar_tabpfn and _is_tabpfn_auth_error(exc):
            _raise_tabpfn_auth_error(exc)
        raise

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
    ignorar: str = Form(None),
    usar_tabpfn: bool = Form(True),
    priorlabs_api_key: str | None = Form(None),
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
            if metodo == "tabpfn" and not usar_tabpfn:
                raise HTTPException(
                    status_code=400,
                    detail="TabPFN foi desativado para esta execução. Ative a opção para usá-lo ou escolha outro método.",
                )
            
            if metodo == "tabpfn" and usar_tabpfn and not priorlabs_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Não é possível rodar o TabPFN sem a chave API. Favor desmarcar o TabPFN ou inserir a API Key.",
                )
            
            try:
                df_imputed = runner.imputar(
                    df=df_imputed,
                    algoritmo=metodo,
                    feature=col,
                    tabpfn_api_key=priorlabs_api_key,
                )
            except RuntimeError as exc:
                if usar_tabpfn and metodo == "tabpfn" and _is_tabpfn_auth_error(exc):
                    _raise_tabpfn_auth_error(exc)
                raise

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