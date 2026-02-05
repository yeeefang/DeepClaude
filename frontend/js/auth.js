/**
 * 身分驗證相關功能
 */
const API_BASE_URL = window.location.origin;
const API_KEY_STORAGE_KEY = 'deepclaude_api_key';
const authPage = document.getElementById('auth-page');
const configPage = document.getElementById('config-page');
const authForm = document.getElementById('auth-form');
const apiKeyInput = document.getElementById('api-key');
const authError = document.getElementById('auth-error');
const logoutBtn = document.getElementById('logout-btn');

function initAuth() {
    const storedApiKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (storedApiKey) {
        verifyApiKey(storedApiKey)
            .then(isValid => { if (isValid) { showConfigPage(); } else { showAuthPage(); } })
            .catch(() => { showAuthPage(); });
    } else { showAuthPage(); }
    authForm.addEventListener('submit', handleAuthFormSubmit);
    logoutBtn.addEventListener('click', handleLogout);
}

async function handleAuthFormSubmit(event) {
    event.preventDefault();
    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) { showAuthError('請輸入 API Key'); return; }
    try {
        const isValid = await verifyApiKey(apiKey);
        if (isValid) { localStorage.setItem(API_KEY_STORAGE_KEY, apiKey); showConfigPage(); }
        else { showAuthError('API Key 無效'); }
    } catch (error) { showAuthError('驗證過程中發生錯誤，請重試'); console.error('驗證 API Key 時發生錯誤:', error); }
}

function handleLogout() { localStorage.removeItem(API_KEY_STORAGE_KEY); showAuthPage(); apiKeyInput.value = ''; }

async function verifyApiKey(apiKey) {
    try {
        const response = await fetch(`${API_BASE_URL}/v1/config`, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        return response.ok;
    } catch (error) { console.error('驗證 API Key 時發生錯誤:', error); return false; }
}

function showAuthPage() { authPage.classList.remove('d-none'); configPage.classList.add('d-none'); authError.classList.add('d-none'); }
function showConfigPage() { authPage.classList.add('d-none'); configPage.classList.remove('d-none'); if (window.Config && typeof window.Config.load === 'function') { window.Config.load(); } }
function showAuthError(message) { authError.textContent = message; authError.classList.remove('d-none'); }
function getCurrentApiKey() { return localStorage.getItem(API_KEY_STORAGE_KEY); }

window.Auth = { init: initAuth, getCurrentApiKey, API_BASE_URL };
