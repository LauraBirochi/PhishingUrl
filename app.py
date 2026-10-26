"""
Aplicação Web Streamlit — Detecção de Phishing com Inteligência Artificial
Projeto Acadêmico: PhiUSIIL Phishing URL Detection
"""

import html
import os
import re
import joblib
import pandas as pd
import streamlit as st
from feature_extractor import extract_features, PageUnreachableError, SSRFSecurityError, FEATURE_COLUMNS

# 1. Configuração da Página
st.set_page_config(
    page_title="LinkSeguro — Classificação Inteligente de URLs",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Carregamento com Cache dos Artefatos de Machine Learning
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Carregando modelos de Inteligência Artificial...")
def load_ml_artifacts():
    model_path = os.path.join(BASE_DIR, "arvore_decisao.joblib")
    encoder_path = os.path.join(BASE_DIR, "codificador.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "Artefatos necessários não encontrados na raiz do projeto. "
            "Certifique-se de que 'arvore_decisao.joblib' e 'codificador.joblib' estão presentes."
        )
        
    modelo = joblib.load(model_path)
    codificador = joblib.load(encoder_path)
    return modelo, codificador

try:
    modelo, codificador = load_ml_artifacts()
except Exception as e:
    st.error(f"Erro ao inicializar o modelo de Machine Learning: {e}")
    st.stop()


# 3. Estilo: o design system vive em style.css e é compartilhado com index.html
FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2"
    "?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');"
)

def inject_custom_styles():
    css_file = os.path.join(BASE_DIR, "style.css")
    custom_css = ""
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            custom_css = f.read()
    st.markdown(f"<style>{FONT_IMPORT}\n{custom_css}</style>", unsafe_allow_html=True)

inject_custom_styles()


def html_block(markup: str) -> str:
    """
    Remove indentação e linhas em branco do HTML antes de enviá-lo ao st.markdown.
    O Markdown encerra um bloco HTML numa linha em branco e trata linhas indentadas
    com 4+ espaços como bloco de código, o que quebrava o cartão de resultado.
    """
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


ICON_SHIELD_OK = """
<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#047857" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
  <path d="m9 12 2 2 4-4"/>
</svg>
"""

ICON_DANGER = """
<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#b91c1c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <circle cx="12" cy="12" r="10"/>
  <line x1="15" y1="9" x2="9" y2="15"/>
  <line x1="9" y1="9" x2="15" y2="15"/>
</svg>
"""

ICON_NOTICE = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
  <line x1="12" y1="9" x2="12" y2="13"/>
  <line x1="12" y1="17" x2="12.01" y2="17"/>
</svg>
"""


# 4. Painel Lateral (Sidebar) com Limitações Técnicas e Detalhes
with st.sidebar:
    st.markdown(
        html_block("""
        <section class="side-section" aria-labelledby="side-about">
          <h2 class="side-title" id="side-about">Sobre o LinkSeguro AI</h2>
          <p class="side-text">
            Aplicação acadêmica para detecção preditiva de URLs fraudulentas (phishing),
            usando um modelo de <strong>Árvore de Decisão</strong> treinado sobre o
            <strong>PhiUSIIL Phishing URL Dataset</strong>.
          </p>
        </section>
        <section class="side-section" aria-labelledby="side-limits">
          <h2 class="side-title" id="side-limits">Limitações da solução</h2>
          <ul class="side-list">
            <li>
              <strong>Acesso HTTP obrigatório.</strong>
              O classificador depende de 28 atributos extraídos do HTML da página.
              Se a URL estiver offline ou inacessível, a predição é suspensa por segurança.
            </li>
            <li>
              <strong>Base de referência (PhiUSIIL).</strong>
              O treinamento usou 235.795 URLs. URLs novas podem apresentar padrões desconhecidos.
            </li>
            <li>
              <strong>Teste controlado vs. ameaças reais.</strong>
              O resultado de 100% no conjunto de teste foi obtido em condições experimentais
              e não garante proteção contra ataques novos (dia zero).
            </li>
          </ul>
        </section>
        <p class="side-caption">Projeto de Introdução à Inteligência Artificial (2026)</p>
        """),
        unsafe_allow_html=True
    )


# 5. Cabeçalho Principal (Brand Header)
st.markdown(
    html_block("""
    <header class="brand-header">
      <div class="logo-shield" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <path d="m9 12 2 2 4-4" />
        </svg>
      </div>
      <h1 class="brand-title">LinkSeguro</h1>
      <p class="brand-subtitle">Classificador de URLs em tempo real baseado em Inteligência Artificial</p>
    </header>
    """),
    unsafe_allow_html=True
)

# 6. Formulário de Consulta e Validação
with st.form("search_form", clear_on_submit=False):
    raw_url = st.text_input(
        "URL para verificação",
        placeholder="Cole ou digite a URL para analisar (ex.: https://exemplo.com)",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Iniciar análise", use_container_width=True)

# 7. Fluxo de Inferência
if submitted:
    clean_url = raw_url.strip()
    
    if not clean_url:
        st.warning("Por favor, informe uma URL antes de iniciar a análise.")
    else:
        # Auto-prefixo de protocolo se necessário
        if not re.match(r"^https?://", clean_url, re.IGNORECASE):
            clean_url = "https://" + clean_url

        with st.spinner("Analisando a URL: realizando requisição segura e extraindo 51 características..."):
            try:
                # Módulo A: Extração das 51 características
                df_features = extract_features(clean_url)
                
                # Módulo B: Transformação Categórica (estritamente transform)
                X_encoded = codificador.transform(df_features)
                
                # Predição do Modelo (0: Phishing, 1: Legítimo)
                pred = int(modelo.predict(X_encoded)[0])
                
                # Cálculo de probabilidade se disponível
                prob_legit = None
                if hasattr(modelo, "predict_proba"):
                    probas = modelo.predict_proba(X_encoded)[0]
                    prob_legit = probas[1] * 100.0 if len(probas) > 1 else (100.0 if pred == 1 else 0.0)

                # 8. Exibição do Resultado Estilizado
                is_https = clean_url.lower().startswith("https://")
                protocol_class = "secure" if is_https else "insecure"
                protocol_text = "HTTPS" if is_https else "HTTP"
                safe_url = html.escape(clean_url, quote=True)
                
                if pred == 1:
                    # Legítimo
                    status_class = "safe"
                    verdict_title = "Link Seguro (Legítimo)"
                    verdict_desc = "O modelo de Árvore de Decisão classificou este endereço como legítimo, compatível com padrões confiáveis."
                    conf_display = f"{prob_legit:.1f}%" if prob_legit is not None else "100%"
                    icon_svg = ICON_SHIELD_OK
                else:
                    # Phishing
                    status_class = "danger"
                    verdict_title = "Alerta: Potencial Phishing"
                    verdict_desc = "CUIDADO: O modelo identificou características associadas a páginas de fraude, clonagem ou phishing."
                    conf_display = f"{(100.0 - prob_legit):.1f}%" if prob_legit is not None else "100%"
                    icon_svg = ICON_DANGER

                # 8. Feedback textual (stAlertContainer) POSICIONADO ACIMA do cartão de resultado
                if pred == 1:
                    st.success(f"**{verdict_title}**: {verdict_desc}")
                else:
                    st.error(f"**{verdict_title}**: {verdict_desc}")

                # 9. Cartão de Resultado Estilizado
                result_card_html = html_block(f"""
                <div class="result-container" role="status">
                  <article class="result-card">
                    <div class="url-analyzed-box">
                      <span class="box-caption">URL analisada</span>
                      <div class="url-display-row">
                        <span class="url-protocol-badge {protocol_class}">{protocol_text}</span>
                        <span class="analyzed-url-text" title="{safe_url}">{safe_url}</span>
                      </div>
                    </div>
                    <div class="verdict-banner {status_class}">
                      <div class="verdict-icon">{icon_svg}</div>
                      <div class="verdict-info">
                        <span class="verdict-label">Classificação do modelo</span>
                        <h2 class="verdict-title">{verdict_title}</h2>
                      </div>
                      <div class="confidence-badge">
                        <span class="confidence-number">{conf_display}</span>
                        <span class="confidence-sub">confiança</span>
                      </div>
                    </div>
                  </article>
                </div>
                """)
                st.markdown(result_card_html, unsafe_allow_html=True)

                # 9. Inspeção Educacional das 51 Características
                with st.expander("Inspecionar as 51 características calculadas para esta URL"):
                    st.markdown("**Valores calculados pelo extrator para inferência:**")
                    
                    # Organiza exibição em abas por categoria
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "URL & Texto (14)", 
                        "Domínio & TLD (6)", 
                        "Conteúdo HTML (28)", 
                        "Índices PhiUSIIL (3)"
                    ])
                    
                    with tab1:
                        cat1_cols = [
                            'URLLength', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL',
                            'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
                            'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
                            'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio'
                        ]
                        st.dataframe(df_features[cat1_cols].T.rename(columns={0: "Valor"}), use_container_width=True)

                    with tab2:
                        cat2_cols = [
                            'DomainLength', 'IsDomainIP', 'TLD', 'TLDLength', 'NoOfSubDomain', 'CharContinuationRate'
                        ]
                        st.dataframe(df_features[cat2_cols].T.rename(columns={0: "Valor"}), use_container_width=True)

                    with tab3:
                        cat3_cols = [
                            'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
                            'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect',
                            'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame',
                            'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
                            'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo',
                            'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
                        ]
                        st.dataframe(df_features[cat3_cols].T.rename(columns={0: "Valor"}), use_container_width=True)

                    with tab4:
                        cat4_cols = ['URLSimilarityIndex', 'TLDLegitimateProb', 'URLCharProb']
                        st.dataframe(df_features[cat4_cols].T.rename(columns={0: "Valor"}), use_container_width=True)

            except SSRFSecurityError as ssrf_err:
                st.markdown(
                    html_block(f"""
                    <div class="notice-box blocked" role="alert">
                      {ICON_NOTICE}
                      <div>
                        <h3>Ação bloqueada por proteção de segurança (anti-SSRF)</h3>
                        <p>{html.escape(str(ssrf_err))}</p>
                      </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
            except PageUnreachableError as reach_err:
                st.markdown(
                    html_block(f"""
                    <div class="notice-box" role="alert">
                      {ICON_NOTICE}
                      <div>
                        <h3>Análise interrompida: página inacessível</h3>
                        <p><strong>Motivo:</strong> {html.escape(str(reach_err))}</p>
                        <p>
                          O modelo de aprendizado de máquina precisa dos dados estruturais do HTML
                          para classificar com fidelidade. Por política do projeto, dados arbitrários
                          não são imputados quando o site está offline ou inacessível.
                        </p>
                      </div>
                    </div>
                    """),
                    unsafe_allow_html=True
                )
            except Exception as unk_err:
                st.error(f"Ocorreu um erro inesperado durante o processamento: {unk_err}")

# 10. Rodapé Informativo (div: a regra de CSS esconde apenas o <footer> nativo do Streamlit)
st.markdown(
    html_block("""
    <div class="footer-note" role="contentinfo" style="margin-top: 48px;">
      <p>A classificação é uma previsão probabilística gerada por Machine Learning e não substitui práticas completas de cibersegurança.</p>
    </div>
    """),
    unsafe_allow_html=True
)
