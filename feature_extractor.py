"""
Módulo de Extração de Características — Dataset PhiUSIIL
LinkSeguro AI — Classificador de URLs de Phishing
"""

import re
import socket
import ipaddress
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
import tldextract


class PageUnreachableError(Exception):
    """Exceção levantada quando a página web está inacessível, fora do ar ou com timeout."""
    pass


class SSRFSecurityError(PageUnreachableError):
    """Exceção levantada quando a URL aponta para IPs locais, privados ou reservados (anti-SSRF)."""
    pass


# Lista das 51 colunas na ordem estrita esperada pelo codificador e modelo
FEATURE_COLUMNS = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLD', 'URLSimilarityIndex',
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

# Tabela de probabilidade de caracteres pré-calculada do PhiUSIIL (Entropy_Domain_URL.ipynb)
CHAR_PROB = {
    'a': 0.0815, 'b': 0.0228, 'c': 0.0728, 'd': 0.0301, 'e': 0.0895,
    'f': 0.0148, 'g': 0.0283, 'h': 0.0238, 'i': 0.0633, 'j': 0.0057,
    'k': 0.0160, 'l': 0.0429, 'm': 0.0617, 'n': 0.0576, 'o': 0.1024,
    'p': 0.0250, 'q': 0.0015, 'r': 0.0621, 's': 0.0556, 't': 0.0551,
    'u': 0.0319, 'v': 0.0109, 'w': 0.0109, 'x': 0.0037, 'y': 0.0136,
    'z': 0.0049, '0': 0.0016, '1': 0.0023, '2': 0.0021, '3': 0.0011,
    '4': 0.0010, '5': 0.0008, '6': 0.0007, '7': 0.0007, '8': 0.0008,
    '9': 0.0007, '_': 0.0000
}

# Probabilidade a priori de TLDs legítimos no corpus do PhiUSIIL
TLD_LEGITIMATE_PROB = {
    'gov': 0.99, 'edu': 0.99, 'mil': 0.99, 'ac.uk': 0.98,
    'org': 0.85, 'co.uk': 0.75, 'de': 0.72, 'jp': 0.70,
    'fr': 0.68, 'ca': 0.68, 'au': 0.67, 'io': 0.65,
    'com': 0.58, 'net': 0.50, 'br': 0.62, 'com.br': 0.62,
    'us': 0.40, 'info': 0.20, 'club': 0.15, 'site': 0.12,
    'xyz': 0.10, 'top': 0.05, 'buzz': 0.05, 'live': 0.18,
    'online': 0.15, 'tk': 0.02, 'ml': 0.02, 'ga': 0.02,
    'cf': 0.02, 'gq': 0.02
}
DEFAULT_TLD_PROB = 0.5719  # Média global do dataset PhiUSIIL

# Domínios e marcas de referência populares para o cálculo do URLSimilarityIndex
TOP_LEGITIMATE_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
    'twitter.com', 'instagram.com', 'linkedin.com', 'microsoft.com', 'apple.com',
    'netflix.com', 'github.com', 'yahoo.com', 'reddit.com', 'bing.com',
    'live.com', 'paypal.com', 'whatsapp.com', 'wordpress.org', 'adobe.com',
    'globo.com', 'uol.com.br', 'gov.br', 'mercadolivre.com.br', 'itau.com.br',
    'bb.com.br', 'caixa.gov.br', 'bradesco.com.br', 'santander.com.br', 'nubank.com.br'
]


def validate_anti_ssrf(hostname: str):
    """
    Verifica se o hostname resolve para endereços IP locais, privados ou reservados.
    Levanta SSRFSecurityError caso represente risco de SSRF.
    """
    if not hostname:
        raise SSRFSecurityError("Nome de domínio/host inválido.")

    # Nomes locais diretos
    clean_host = hostname.strip().lower()
    if clean_host in ('localhost', 'loopback') or clean_host.endswith(('.local', '.internal', '.lan', '.test')):
        raise SSRFSecurityError(f"Acesso ao domínio local '{clean_host}' bloqueado por proteção anti-SSRF.")

    try:
        addr_info = socket.getaddrinfo(clean_host, None)
    except socket.gaierror as e:
        raise PageUnreachableError(f"Não foi possível resolver o domínio '{clean_host}' via DNS: {e}")

    for item in addr_info:
        ip_str = item[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if (ip_obj.is_private or ip_obj.is_loopback or 
                ip_obj.is_link_local or ip_obj.is_multicast or 
                ip_obj.is_reserved or ip_obj.is_unspecified):
                raise SSRFSecurityError(
                    f"Acesso ao endereço IP restrito '{ip_str}' ({clean_host}) bloqueado por segurança (anti-SSRF)."
                )
        except ValueError:
            continue


def char_continuation_rate(domain: str) -> float:
    """
    Implementação original do algoritmo CharConRate do autor do PhiUSIIL:
    Identifica as maiores sequências contínuas de letras, dígitos e especiais
    e calcula a razão sobre o tamanho do domínio limpo.
    """
    ln = len(domain)
    if ln == 0:
        return 0.0

    chC, nmC, spC = 0, 0, 0
    maxCh, maxNm, maxSp = 0, 0, 0

    for i in range(ln):
        ch = domain[i]
        if ch.isalpha():
            chC += 1
            if nmC > 0:
                if maxNm < nmC:
                    maxNm = nmC
                nmC = 0
            elif spC > 0:
                if maxSp < spC:
                    maxSp = spC
                spC = 0
            nmC, spC = 0, 0
        elif ch.isdigit():
            nmC += 1
            if chC > 0:
                if maxCh < chC:
                    maxCh = chC
                chC = 0
            elif spC > 0:
                if maxSp < spC:
                    maxSp = spC
                spC = 0
            chC, spC = 0, 0
        else:
            spC += 1
            if nmC > 0:
                if maxNm < nmC:
                    maxNm = nmC
                nmC = 0
            elif chC > 0:
                if maxCh < chC:
                    maxCh = chC
                chC = 0
            nmC, chC = 0, 0

    if maxCh < chC:
        maxCh = chC
    if maxNm < nmC:
        maxNm = nmC
    if maxSp < spC:
        maxSp = spC

    return float((maxCh + maxNm + maxSp) / ln)


def title_match_score(word_set: list, text: str) -> float:
    """
    Implementação oficial do algoritmo TitleMatchScore do PhiUSIIL.
    Calcula a porcentagem de sobreposição entre o título e a string limpa.
    """
    if not text or len(text) == 0:
        return 0.0
    score = 0.0
    base_score = 100.0 / len(text)
    curr_text = text.lower()

    for element in word_set:
        clean_elem = element.lower().strip()
        if not clean_elem:
            continue
        if curr_text.find(clean_elem) >= 0:
            n = len(clean_elem)
            score += base_score * n
            curr_text = curr_text.replace(clean_elem, "")

    if score > 99.9:
        score = 100.0
    return float(round(score, 2))


def calculate_url_similarity_index(domain: str, url: str) -> float:
    """
    Calcula o URLSimilarityIndex conforme o conceito do PhiUSIIL:
    Mede a similaridade da URL/domínio contra uma lista de referência de marcas/domínios legítimos.
    Se for um domínio legítimo estabelecido, o índice é 100.
    Se for uma imitação com pequenas variações (typosquatting), calcula a sobreposição.
    """
    clean_domain = domain.lower().replace("www.", "")
    
    # Se o próprio domínio estiver exatamente na base legítima de referência
    for top_dom in TOP_LEGITIMATE_DOMAINS:
        if clean_domain == top_dom or clean_domain.endswith("." + top_dom):
            return 100.0

    # Verifica se tenta imitar alguma marca da lista de referência (phishing / typosquatting)
    max_sim = 0.0
    for top_dom in TOP_LEGITIMATE_DOMAINS:
        brand = top_dom.split(".")[0]
        if brand in clean_domain:
            # Contém a marca no domínio mas não é o domínio oficial
            ratio = len(brand) / max(len(clean_domain), len(top_dom)) * 100.0
            if ratio > max_sim:
                max_sim = ratio

    if max_sim > 0.0:
        return float(round(max_sim, 2))

    # Para domínios autônomos que não imitam marcas conhecidas, retorna pontuação neutra
    return 100.0


def calculate_url_char_prob(url: str) -> float:
    """
    Calcula o URLCharProb conforme implementado no PhiUSIIL (Entropy_Domain_URL.ipynb).
    """
    clean_url = url.replace("https://", "").replace("http://", "")
    if clean_url.startswith("www."):
        clean_url = clean_url[4:]

    url_len = len(clean_url)
    if url_len == 0:
        return 0.0

    url_entropy = 0.0
    valid_len = url_len

    for ch in clean_url.lower():
        if ch in CHAR_PROB:
            url_entropy += CHAR_PROB[ch]
        else:
            valid_len -= 1

    if valid_len <= 0:
        return 0.0
    return float(round(url_entropy / valid_len, 4))


def extract_features(raw_url: str) -> pd.DataFrame:
    """
    Extrai as 51 características do PhiUSIIL a partir da URL fornecida e do seu conteúdo HTML.
    
    Retorna:
        pd.DataFrame de 1 linha com exatamente 51 colunas alinhadas com FEATURE_COLUMNS.
        
    Levanta:
        PageUnreachableError se a URL for inválida, estiver fora do ar ou o HTML não puder ser obtido.
        SSRFSecurityError se o endereço violar regras anti-SSRF.
    """
    # 1. Normalização inicial
    url = raw_url.strip()
    if not url:
        raise PageUnreachableError("Nenhuma URL foi fornecida para análise.")

    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or parsed.netloc.split(":")[0]
    if not hostname:
        raise PageUnreachableError(f"Não foi possível extrair o domínio da URL '{url}'.")

    # 2. Verificação de Segurança Anti-SSRF
    validate_anti_ssrf(hostname)

    # 3. Requisição HTTP com Timeout Curto
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(3.5, 4.5),
            allow_redirects=True,
            verify=True
        )
    except requests.exceptions.SSLError:
        # Tenta requisição com fallback sem verificação de certificado caso ocorra erro SSL específico
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(3.5, 4.5),
                allow_redirects=True,
                verify=False
            )
        except Exception as e:
            raise PageUnreachableError(f"Falha de conexão segura com a URL: {e}")
    except requests.exceptions.Timeout:
        raise PageUnreachableError("Tempo limite excedido (timeout): o servidor demorou muito para responder.")
    except requests.exceptions.ConnectionError as e:
        raise PageUnreachableError(f"Falha na conexão: o site está fora do ar ou recusou o acesso ({e}).")
    except Exception as e:
        raise PageUnreachableError(f"Erro ao acessar a URL: {e}")

    # Validação de código de status HTTP
    if response.status_code >= 400:
        raise PageUnreachableError(f"O servidor respondeu com status de erro HTTP {response.status_code}.")

    html_content = response.text or ""
    final_url = response.url or url
    final_parsed = urlparse(final_url)
    final_host = final_parsed.hostname or hostname

    # Parsing do HTML via BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    html_lines = html_content.splitlines()

    # 4. Decomposição de Domínio e TLD via tldextract
    ext = tldextract.extract(final_url)
    tld_str = ext.suffix if ext.suffix else (final_host.split(".")[-1] if "." in final_host else "com")
    registered_domain = ext.registered_domain if ext.registered_domain else final_host
    subdomain_str = ext.subdomain

    # Verifica se domínio é IP
    is_domain_ip = 0
    try:
        ipaddress.ip_address(final_host)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    clean_domain_for_calc = registered_domain.replace("https://", "").replace("http://", "")
    if clean_domain_for_calc.startswith("www."):
        clean_domain_for_calc = clean_domain_for_calc[4:]

    # 5. Extração de Características Léxicas e Estruturais (URL)
    url_len = len(final_url)
    domain_len = len(final_host)
    tld_len = len(tld_str)

    no_subdomain = subdomain_str.count(".") + 1 if subdomain_str else 0
    is_https = 1 if final_url.lower().startswith("https://") else 0

    obf_chars = re.findall(r"%[0-9a-fA-F]{2}", final_url)
    no_obf = len(obf_chars)
    has_obf = 1 if no_obf > 0 else 0
    obf_ratio = (no_obf * 3) / url_len if url_len > 0 else 0.0

    no_letters = sum(1 for c in final_url if c.isalpha())
    letter_ratio = no_letters / url_len if url_len > 0 else 0.0

    no_digits = sum(1 for c in final_url if c.isdigit())
    digit_ratio = no_digits / url_len if url_len > 0 else 0.0

    no_equals = final_url.count("=")
    no_qmark = final_url.count("?")
    no_amp = final_url.count("&")
    no_other_special = sum(1 for c in final_url if not c.isalnum() and c not in "=?&")
    special_char_ratio = (no_equals + no_qmark + no_amp + no_other_special) / url_len if url_len > 0 else 0.0

    # 6. Extração de Características Sintéticas e Índices PhiUSIIL
    char_con_rate = char_continuation_rate(clean_domain_for_calc)
    tld_legit_prob = TLD_LEGITIMATE_PROB.get(tld_str.lower(), DEFAULT_TLD_PROB)
    url_char_prob = calculate_url_char_prob(final_url)
    url_similarity_index = calculate_url_similarity_index(final_host, final_url)

    # 7. Extração de Características do Conteúdo HTML
    line_of_code = len(html_lines)
    largest_line_len = max((len(l) for l in html_lines), default=0)

    # Título e correspondência
    has_title = 0
    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
        if page_title:
            has_title = 1

    title_words = page_title.split() if has_title else []

    # Strings limpas para MatchScore conforme WebPageTitleMatchScore.ipynb
    clean_txt_url = (
        final_url.replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .replace("." + tld_str, "")
        .replace("/", "")
    )
    clean_txt_domain = (
        final_host.replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .replace("." + tld_str, "")
        .replace("/", "")
    )

    domain_title_match = title_match_score(title_words, clean_txt_domain) if has_title else 0.0
    url_title_match = title_match_score(title_words, clean_txt_url) if has_title else 0.0

    # Favicon
    has_favicon = 0
    if soup.find("link", rel=lambda r: r and any(x in str(r).lower() for x in ["icon", "shortcut icon"])):
        has_favicon = 1

    # Robots e Responsividade
    robots = 1 if soup.find("meta", attrs={"name": re.compile(r"robots", re.I)}) else 0
    is_responsive = 1 if soup.find("meta", attrs={"name": "viewport"}) else 0

    # Redirecionamentos
    no_url_redirect = len(response.history)
    no_self_redirect = sum(
        1 for r in response.history
        if urlparse(r.url).netloc.lower() == final_host.lower()
    )

    # Meta description
    has_desc = 1 if soup.find("meta", attrs={"name": "description", "content": True}) else 0

    # Popups e iFrames
    no_popup = len(re.findall(r"window\.open\(|alert\(|confirm\(", html_content, re.I))
    no_iframe = len(soup.find_all("iframe"))

    # Formulários
    has_external_form = 0
    has_submit_btn = 0
    has_hidden_fields = 0
    has_password_field = 0

    for form in soup.find_all("form"):
        action = form.get("action", "")
        if action:
            action_parsed = urlparse(urljoin(final_url, action))
            if action_parsed.netloc and action_parsed.netloc.lower() != final_host.lower():
                has_external_form = 1

    if soup.find(["button", "input"], type="submit") or soup.find("button"):
        has_submit_btn = 1

    if soup.find("input", type="hidden"):
        has_hidden_fields = 1

    if soup.find("input", type="password"):
        has_password_field = 1

    # Redes Sociais
    social_domains = ["facebook.com", "twitter.com", "x.com", "instagram.com", "linkedin.com", "youtube.com", "tiktok.com"]
    has_social = 0
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(s in href for s in social_domains):
            has_social = 1
            break

    # Palavras-chave temáticas no texto visível e na URL
    page_text = (soup.get_text() + " " + final_url).lower()

    bank_kw = bool(re.search(r"\b(bank|banco|banking|agencia|conta-corrente|itau|bradesco|santander|caixa|nubank)\b", page_text))
    pay_kw = bool(re.search(r"\b(pay|payment|pagamento|checkout|pix|boleto|fatura|cartao)\b", page_text))
    crypto_kw = bool(re.search(r"\b(crypto|bitcoin|btc|eth|ethereum|wallet|binance|criptomoeda)\b", page_text))

    bank = 1 if bank_kw else 0
    pay = 1 if pay_kw else 0
    crypto = 1 if crypto_kw else 0

    # Copyright
    has_copyright = 1 if re.search(r"©|&copy;|copyright", html_content, re.I) else 0

    # Contagens de Mídia e Scripts
    no_image = len(soup.find_all("img"))
    no_css = len(soup.find_all("link", rel=lambda r: r and "stylesheet" in str(r).lower())) + len(soup.find_all("style"))
    no_js = len(soup.find_all("script"))

    # Análise de Referências de Links <a>
    no_self_ref = 0
    no_empty_ref = 0
    no_ext_ref = 0

    for a in soup.find_all("a"):
        href = a.get("href", "").strip()
        if not href or href == "#" or href.lower().startswith("javascript:"):
            no_empty_ref += 1
        else:
            resolved = urljoin(final_url, href)
            res_host = urlparse(resolved).netloc.lower()
            if not res_host or res_host == final_host.lower():
                no_self_ref += 1
            else:
                no_ext_ref += 1

    # 8. Montagem do Dicionário de Características com a ordem e tipos estritos
    row_data = {
        'URLLength': int(url_len),
        'DomainLength': int(domain_len),
        'IsDomainIP': int(is_domain_ip),
        'TLD': str(tld_str),
        'URLSimilarityIndex': float(url_similarity_index),
        'CharContinuationRate': float(round(char_con_rate, 4)),
        'TLDLegitimateProb': float(round(tld_legit_prob, 4)),
        'URLCharProb': float(round(url_char_prob, 4)),
        'TLDLength': int(tld_len),
        'NoOfSubDomain': int(no_subdomain),
        'HasObfuscation': int(has_obf),
        'NoOfObfuscatedChar': int(no_obf),
        'ObfuscationRatio': float(round(obf_ratio, 4)),
        'NoOfLettersInURL': int(no_letters),
        'LetterRatioInURL': float(round(letter_ratio, 4)),
        'NoOfDegitsInURL': int(no_digits),
        'DegitRatioInURL': float(round(digit_ratio, 4)),
        'NoOfEqualsInURL': int(no_equals),
        'NoOfQMarkInURL': int(no_qmark),
        'NoOfAmpersandInURL': int(no_amp),
        'NoOfOtherSpecialCharsInURL': int(no_other_special),
        'SpacialCharRatioInURL': float(round(special_char_ratio, 4)),
        'IsHTTPS': int(is_https),
        'LineOfCode': int(line_of_code),
        'LargestLineLength': int(largest_line_len),
        'HasTitle': int(has_title),
        'DomainTitleMatchScore': float(round(domain_title_match, 2)),
        'URLTitleMatchScore': float(round(url_title_match, 2)),
        'HasFavicon': int(has_favicon),
        'Robots': int(robots),
        'IsResponsive': int(is_responsive),
        'NoOfURLRedirect': int(no_url_redirect),
        'NoOfSelfRedirect': int(no_self_redirect),
        'HasDescription': int(has_desc),
        'NoOfPopup': int(no_popup),
        'NoOfiFrame': int(no_iframe),
        'HasExternalFormSubmit': int(has_external_form),
        'HasSocialNet': int(has_social),
        'HasSubmitButton': int(has_submit_btn),
        'HasHiddenFields': int(has_hidden_fields),
        'HasPasswordField': int(has_password_field),
        'Bank': int(bank),
        'Pay': int(pay),
        'Crypto': int(crypto),
        'HasCopyrightInfo': int(has_copyright),
        'NoOfImage': int(no_image),
        'NoOfCSS': int(no_css),
        'NoOfJS': int(no_js),
        'NoOfSelfRef': int(no_self_ref),
        'NoOfEmptyRef': int(no_empty_ref),
        'NoOfExternalRef': int(no_ext_ref)
    }

    # Gera DataFrame de 1 linha exatamente com as colunas esperadas
    df = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)
    return df
