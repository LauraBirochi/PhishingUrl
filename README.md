# Detecção em Tempo Real de URLs de Phishing com IA

## 🔗 Link Público da Aplicação
* **Deploy no Streamlit:** [https://phishingurl-4pre7ejrqpsbjip5gt4dkc.streamlit.app/](https://phishingurl-4pre7ejrqpsbjip5gt4dkc.streamlit.app/)

## 👥 Integrantes do Grupo
* Laura Birochi (CT3037398)
* Maria Clara Rosa Hespanhol (CT3037657)
* Polayne da Silva Bastos (CT3038203)

## 📂 Repositório GitHub 
* [https://github.com/LauraBirochi/PhishingUrl](https://github.com/LauraBirochi/PhishingUrl)
## 💻 Instruções de Execução Local

1. **Criar e ativar o ambiente virtual:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Executar a aplicação:**
   ```bash
   streamlit run app.py
   ```

4. **Acessar no navegador:**
   * A aplicação abrirá automaticamente no endereço: `http://localhost:8501`

---


## 🔄 Reproduzindo o Treinamento do Modelo (`treinamento.py`)

Caso deseje reexecutar todo o processo de preparação, treino e validação a partir da base bruta do **PhiUSIIL**:

1. Baixe o dataset oficial (`PhiUSIIL_Phishing_URL_Dataset.csv`) através do [Repositório UCI](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) e posicione-o na raiz do projeto.
2. Execute o script de treinamento:
   ```bash
   python treinamento.py
   ```
3. O script exibirá as métricas no terminal (Acurácia, F1-Score, Matriz de Confusão e Relatório de Classificação) e atualizará os artefatos `arvore_decisao.joblib` e `codificador.joblib`.

