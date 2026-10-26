"""
Script de Treinamento e Avaliação do Modelo — LinkSeguro AI
Dataset de Referência: PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository)
Algoritmo: Árvore de Decisão (DecisionTreeClassifier)
Pipeline: Pré-processamento com ColumnTransformer + OneHotEncoder

Este script reproduz integralmente a etapa de modelagem executada originalmente
no Google Colab, realizando o pré-processamento dos dados, treinamento,
avaliação de métricas de desempenho e serialização dos artefatos (.joblib).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Definição das 51 Características Extraídas e Alvo
CATEGORICAL_FEATURES = ['TLD']
NUMERICAL_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
    'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
    'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
    'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
    'NoOfEmptyRef', 'NoOfExternalRef'
]

ALL_FEATURES = ['URLLength', 'DomainLength', 'IsDomainIP', 'TLD'] + NUMERICAL_FEATURES[3:]
TARGET_COLUMN = 'label'  # 1 = Legítimo, 0 = Phishing


def carregar_dados(caminho_csv: str = "PhiUSIIL_Phishing_URL_Dataset.csv") -> pd.DataFrame:
    """
    Carrega o dataset PhiUSIIL. Caso o arquivo não esteja disponível localmente,
    orienta o usuário com as instruções de download oficial da UCI/Kaggle.
    """
    if os.path.exists(caminho_csv):
        print(f"[*] Carregando base de dados: {caminho_csv}...")
        df = pd.read_csv(caminho_csv)
        print(f"[+] Base carregada com sucesso: {df.shape[0]} amostras e {df.shape[1]} colunas.")
        return df
    else:
        print("\n" + "=" * 70)
        print("AVISO: Arquivo do dataset PhiUSIIL não encontrado localmente.")
        print("Para treinar com os dados brutos completos (235.795 URLs):")
        print("1. Baixe o dataset em: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset")
        print(f"2. Salve o arquivo CSV com o nome '{caminho_csv}' na raiz deste projeto.")
        print("=" * 70 + "\n")
        return None


def construir_preprocessador() -> ColumnTransformer:
    """
    Constrói o ColumnTransformer com OneHotEncoder para a coluna categórica 'TLD'
    e passthrough para as 50 colunas numéricas/binárias.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor


def treinar_modelo(df: pd.DataFrame, salvar_artefatos: bool = True):
    """
    Executa a divisão treino/teste, ajuste do codificador, treinamento da
    Árvore de Decisão e serialização dos artefatos em .joblib.
    """
    # Verificação de colunas necessárias
    colunas_presentes = [c for c in ALL_FEATURES if c in df.columns]
    if len(colunas_presentes) < len(ALL_FEATURES):
        faltantes = set(ALL_FEATURES) - set(colunas_presentes)
        raise ValueError(f"Colunas ausentes no DataFrame: {faltantes}")

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # Divisão em 80% Treino e 20% Teste estratificado
    print("[*] Dividindo dados em 80% treino e 20% teste (estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 1. Ajuste e Transformação pelo Pré-processador
    print("[*] Ajustando pré-processador ColumnTransformer (OneHotEncoder no TLD)...")
    preprocessor = construir_preprocessador()
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    print(f"[+] Dimensão da matriz de entrada pós-codificação: {X_train_encoded.shape[1]} atributos.")

    # 2. Treinamento da Árvore de Decisão
    print("[*] Treinando modelo DecisionTreeClassifier (random_state=42)...")
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train_encoded, y_train)
    print("[+] Treinamento concluído!")

    # 3. Avaliação no Conjunto de Teste
    print("\n" + "=" * 50)
    print("           RELATÓRIO DE AVALIAÇÃO (TESTE)")
    print("=" * 50)
    y_pred = modelo.predict(X_test_encoded)

    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='binary')
    revocacao = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}")
    print(f"Recall:    {revocacao:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação Completo:")
    print(classification_report(y_test, y_pred, target_names=['Phishing (0)', 'Legítimo (1)']))

    # 4. Exportação dos Artefatos Serializados
    if salvar_artefatos:
        print("[*] Serializando modelos para produção com joblib...")
        joblib.dump(modelo, "arvore_decisao.joblib")
        joblib.dump(preprocessor, "codificador.joblib")
        print("[+] 'arvore_decisao.joblib' e 'codificador.joblib' gerados com sucesso na raiz do projeto!")

    return modelo, preprocessor


def main():
    print("=" * 70)
    print("  LinkSeguro AI — Pipeline de Treinamento e Avaliação (Offline)")
    print("=" * 70)

    caminho = "PhiUSIIL_Phishing_URL_Dataset.csv"
    if len(sys.argv) > 1:
        caminho = sys.argv[1]

    df = carregar_dados(caminho)
    if df is not None:
        treinar_modelo(df, salvar_artefatos=True)
    else:
        # Se os artefatos já existirem na pasta, apenas valida a integridade
        if os.path.exists("arvore_decisao.joblib") and os.path.exists("codificador.joblib"):
            print("[OK] Artefatos binarios 'arvore_decisao.joblib' e 'codificador.joblib' estao salvos e validos na pasta.")
            m = joblib.load("arvore_decisao.joblib")
            c = joblib.load("codificador.joblib")
            print(f"[OK] Verificacao concluida: Modelo {type(m).__name__} com {m.n_features_in_} features de entrada.")
        else:
            print("[AVISO] Insira o dataset para executar o treinamento inicial.")


if __name__ == "__main__":
    main()
