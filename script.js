/**
 * LinkSeguro — Frontend Logic
 * Pure Vanilla JavaScript: input interactions, validation, smooth views transition, mock classification
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Formulário e Busca
  const urlForm = document.getElementById('url-form');
  const urlInput = document.getElementById('url-input');
  const clearBtn = document.getElementById('clear-btn');
  const searchBarContainer = document.getElementById('search-bar-container');
  const validationMsg = document.getElementById('validation-msg');
  const validationText = document.getElementById('validation-text');

  // Minitutorial Offcanvas Elements
  const openTutorialBtn = document.getElementById('open-tutorial-btn');
  const closeTutorialBtn = document.getElementById('close-tutorial-btn');
  const dismissTutorialBtn = document.getElementById('dismiss-tutorial-btn');
  const tutorialOverlay = document.getElementById('tutorial-overlay');
  const tutorialOffcanvas = document.getElementById('tutorial-offcanvas');

  // Telas / Views
  const searchView = document.getElementById('search-view');
  const loadingView = document.getElementById('loading-view');
  const resultView = document.getElementById('result-view');

  // Elementos da Tela de Carregamento
  const loadingTargetUrl = document.getElementById('loading-target-url');
  const progressBar = document.getElementById('progress-bar');
  const loadingStepText = document.getElementById('loading-step-text');

  // Elementos do Cartão de Resultado
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const confidenceNumber = document.getElementById('confidence-number');
  const protocolBadge = document.getElementById('protocol-badge');
  const analyzedUrlText = document.getElementById('analyzed-url-text');
  const backBtn = document.getElementById('back-btn');


  // =========================================================================
  // 1. Minitutorial Offcanvas (Abrir, Fechar, Recolher e Voltar)
  // =========================================================================
  function getTutorialFocusable() {
    return Array.from(
      tutorialOffcanvas.querySelectorAll('button, [href], input, [tabindex]:not([tabindex="-1"])')
    );
  }

  function openTutorial() {
    tutorialOverlay.classList.add('active');
    tutorialOffcanvas.classList.add('active');
    tutorialOverlay.setAttribute('aria-hidden', 'false');
    tutorialOffcanvas.setAttribute('aria-hidden', 'false');
    openTutorialBtn.setAttribute('aria-expanded', 'true');
    document.body.style.overflow = 'hidden';
    closeTutorialBtn.focus();
  }

  function closeTutorial() {
    tutorialOverlay.classList.remove('active');
    tutorialOffcanvas.classList.remove('active');
    tutorialOverlay.setAttribute('aria-hidden', 'true');
    tutorialOffcanvas.setAttribute('aria-hidden', 'true');
    openTutorialBtn.setAttribute('aria-expanded', 'false');
    document.body.style.overflow = '';
    openTutorialBtn.focus();
  }

  if (openTutorialBtn) {
    openTutorialBtn.addEventListener('click', openTutorial);
  }

  if (closeTutorialBtn) {
    closeTutorialBtn.addEventListener('click', closeTutorial);
  }

  if (dismissTutorialBtn) {
    dismissTutorialBtn.addEventListener('click', closeTutorial);
  }

  if (tutorialOverlay) {
    tutorialOverlay.addEventListener('click', closeTutorial);
  }

  document.addEventListener('keydown', (e) => {
    if (!tutorialOffcanvas.classList.contains('active')) return;

    if (e.key === 'Escape') {
      closeTutorial();
      return;
    }

    // Mantém o foco do teclado dentro do diálogo enquanto ele está aberto
    if (e.key === 'Tab') {
      const focusable = getTutorialFocusable();
      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  });


  // =========================================================================
  // 2. Manipulação do Campo de Entrada (o foco visual é feito por :focus-within no CSS)
  // =========================================================================
  urlInput.addEventListener('input', () => {
    toggleClearButton();
    hideValidation();
  });

  function toggleClearButton() {
    if (urlInput.value.trim().length > 0) {
      clearBtn.classList.add('visible');
    } else {
      clearBtn.classList.remove('visible');
    }
  }

  clearBtn.addEventListener('click', () => {
    urlInput.value = '';
    toggleClearButton();
    hideValidation();
    urlInput.focus();
  });


  // =========================================================================
  // 3. Submissão e Validação da URL
  // =========================================================================
  urlForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const rawUrl = urlInput.value.trim();

    if (!rawUrl) {
      showValidation('Cole ou digite uma URL para iniciar a análise.');
      return;
    }

    if (!isValidUrlFormat(rawUrl)) {
      showValidation('Formato inválido. Digite um endereço como "exemplo.com" ou "https://..."');
      return;
    }

    hideValidation();
    startAnalysis(rawUrl);
  });

  function isValidUrlFormat(string) {
    const pattern = /^(https?:\/\/)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(:\d+)?(\/.*)?$/i;
    return pattern.test(string);
  }

  function showValidation(message) {
    validationText.textContent = message;
    validationMsg.classList.add('active');
    searchBarContainer.classList.add('has-error');

    setTimeout(() => {
      searchBarContainer.classList.remove('has-error');
    }, 450);

    urlInput.focus();
  }

  function hideValidation() {
    validationMsg.classList.remove('active');
    searchBarContainer.classList.remove('has-error');
  }


  // =========================================================================
  // 4. Fluxo e Transição de Telas
  // =========================================================================
  function switchView(targetView) {
    [searchView, loadingView, resultView].forEach(view => {
      view.classList.remove('active');
      view.setAttribute('aria-hidden', 'true');
    });

    targetView.classList.add('active');
    targetView.setAttribute('aria-hidden', 'false');
  }

  function startAnalysis(url) {
    let normalizedUrl = url;
    if (!/^https?:\/\//i.test(normalizedUrl)) {
      normalizedUrl = `https://${normalizedUrl}`;
    }

    loadingTargetUrl.textContent = normalizedUrl;
    analyzedUrlText.textContent = normalizedUrl;

    switchView(loadingView);
    document.getElementById('loading-title').focus({ preventScroll: true });
    runAnalysisSimulation(normalizedUrl);
  }

  function runAnalysisSimulation(url) {
    const steps = [
      { progress: 20, text: 'Resolvendo DNS e verificando proteção anti-SSRF...' },
      { progress: 50, text: 'Inspecionando estrutura léxica e TLD do domínio...' },
      { progress: 75, text: 'Extraindo 28 características do conteúdo HTML...' },
      { progress: 95, text: 'Processando vetor de 717 variáveis no modelo de Árvore de Decisão...' },
      { progress: 100, text: 'Classificação concluída!' }
    ];

    let currentStep = 0;
    progressBar.style.width = '0%';

    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        const item = steps[currentStep];
        progressBar.style.width = `${item.progress}%`;
        loadingStepText.textContent = item.text;
        currentStep++;
      } else {
        clearInterval(interval);
        setTimeout(() => {
          showResult(url);
        }, 350);
      }
    }, 420);
  }


  // =========================================================================
  // 5. Exibição e Semântica do Veredito
  // =========================================================================
  function showResult(url) {
    switchView(resultView);
    verdictTitle.focus({ preventScroll: true });

    const isHttp = url.startsWith('http://');
    const lowerUrl = url.toLowerCase();

    const isPhishingSample = lowerUrl.includes('phishing') ||
      lowerUrl.includes('seguranca-atualizar') ||
      lowerUrl.includes('verify-account') ||
      lowerUrl.includes('recadastramento') ||
      lowerUrl.includes('.tk') ||
      lowerUrl.includes('.xyz') ||
      lowerUrl.includes('login-alert');

    let status = 'safe';
    let confidence = 98.4;

    if (isPhishingSample) {
      status = 'danger';
      confidence = 99.6;
    } else if (isHttp) {
      status = 'warning';
      confidence = 91.2;
    } else {
      status = 'safe';
      confidence = 99.1;
    }

    if (isHttp) {
      protocolBadge.textContent = 'HTTP';
      protocolBadge.className = 'url-protocol-badge insecure';
    } else {
      protocolBadge.textContent = 'HTTPS';
      protocolBadge.className = 'url-protocol-badge secure';
    }

    verdictBanner.className = `verdict-banner ${status}`;
    confidenceNumber.textContent = `${confidence}%`;

    if (status === 'safe') {
      verdictTitle.textContent = 'Link Seguro';
      verdictIcon.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      `;
    } else if (status === 'warning') {
      verdictTitle.textContent = 'Atenção: Suspeito';
      verdictIcon.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      `;
    } else {
      verdictTitle.textContent = 'Alerta de Phishing';
      verdictIcon.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
      `;
    }
  }

  // Voltar para a Tela de Busca
  backBtn.addEventListener('click', () => {
    switchView(searchView);
    urlInput.value = '';
    toggleClearButton();
    urlInput.focus();
  });
});
