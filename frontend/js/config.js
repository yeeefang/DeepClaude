/**
 * 設定管理相關功能
 */
let configData = {
    reasoner_models: {}, target_models: {}, composite_models: {},
    proxy: { proxy_open: false, proxy_address: "" },
    system: { allow_origins: ["*"], log_level: "INFO", api_key: "123456", save_deepseek_tokens: false, save_deepseek_tokens_max_tokens: 5, max_reasoning_tokens: 0 }
};

const addModelModal = new bootstrap.Modal(document.getElementById('add-model-modal'));
const confirmDeleteModal = new bootstrap.Modal(document.getElementById('confirm-delete-modal'));
const importConfigModal = new bootstrap.Modal(document.getElementById('import-config-modal'));
const deleteModelNameSpan = document.getElementById('delete-model-name');
const confirmDeleteBtn = document.getElementById('confirm-delete-btn');
const addModelForm = document.getElementById('add-model-form');
const addModelFields = document.getElementById('add-model-fields');
const confirmAddModelBtn = document.getElementById('confirm-add-model');
const addModelTitle = document.getElementById('add-model-title');
const exportConfigBtn = document.getElementById('export-config-btn');
const importConfigBtn = document.getElementById('import-config-btn');
const configFileInput = document.getElementById('config-file-input');
const configPreview = document.getElementById('config-preview');
const configPreviewContent = document.getElementById('config-preview-content');
const confirmImportBtn = document.getElementById('confirm-import-btn');
const reasonerModelsContainer = document.getElementById('reasoner-models-container');
const targetModelsContainer = document.getElementById('target-models-container');
const compositeModelsContainer = document.getElementById('composite-models-container');
const addReasonerModelBtn = document.getElementById('add-reasoner-model-btn');
const addTargetModelBtn = document.getElementById('add-target-model-btn');
const addCompositeModelBtn = document.getElementById('add-composite-model-btn');
const saveAllBtn = document.getElementById('save-all-btn');
const saveProxyBtn = document.getElementById('save-proxy-btn');
const saveSystemBtn = document.getElementById('save-system-btn');
const proxyOpenSwitch = document.getElementById('proxy-open');
const proxyAddressInput = document.getElementById('proxy-address');
const allowOriginsContainer = document.getElementById('allow-origins-container');
const addOriginBtn = document.getElementById('add-origin-btn');
const logLevelSelect = document.getElementById('log-level');
const systemApiKeyInput = document.getElementById('system-api-key');
let selectedConfigData = null;

function initConfig() {
    addReasonerModelBtn.addEventListener('click', () => showAddModelModal('reasoner'));
    addTargetModelBtn.addEventListener('click', () => showAddModelModal('target'));
    addCompositeModelBtn.addEventListener('click', () => showAddModelModal('composite'));
    saveAllBtn.addEventListener('click', saveAllConfigurations);
    saveProxyBtn.addEventListener('click', saveProxySettings);
    saveSystemBtn.addEventListener('click', saveSystemSettings);
    exportConfigBtn.addEventListener('click', exportConfiguration);
    importConfigBtn.addEventListener('click', () => importConfigModal.show());
    configFileInput.addEventListener('change', handleConfigFileSelect);
    confirmImportBtn.addEventListener('click', handleConfigImport);
    addOriginBtn.addEventListener('click', addAllowOriginInput);
    confirmAddModelBtn.addEventListener('click', handleAddModel);
    confirmDeleteBtn.addEventListener('click', handleDeleteModel);
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            const targetId = this.getAttribute('href').substring(1);
            document.querySelectorAll('.tab-pane').forEach(pane => { pane.classList.remove('show', 'active'); });
            document.getElementById(targetId).classList.add('show', 'active');
        });
    });
}

async function loadConfigData() {
    try {
        showToast('正在載入設定資料...', 'info');
        const apiKey = Auth.getCurrentApiKey();
        const response = await fetch(`${Auth.API_BASE_URL}/v1/config`, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        if (!response.ok) { throw new Error('載入設定資料失敗'); }
        configData = await response.json();
        if (!configData.system) {
            configData.system = { allow_origins: ["*"], log_level: "INFO", api_key: "123456", save_deepseek_tokens: false, save_deepseek_tokens_max_tokens: 5, max_reasoning_tokens: 0 };
        }
        if (!configData.system.hasOwnProperty('save_deepseek_tokens')) { configData.system.save_deepseek_tokens = false; }
        if (!configData.system.hasOwnProperty('save_deepseek_tokens_max_tokens')) { configData.system.save_deepseek_tokens_max_tokens = 5; }
        if (!configData.system.hasOwnProperty('max_reasoning_tokens')) { configData.system.max_reasoning_tokens = 0; }
        renderModels();
        renderProxySettings();
        renderSystemSettings();
        showToast('設定資料載入成功', 'success');
    } catch (error) {
        console.error('載入設定資料時發生錯誤:', error);
        showToast('載入設定資料失敗: ' + error.message, 'danger');
    }
}

function renderModels() {
    reasonerModelsContainer.innerHTML = '';
    targetModelsContainer.innerHTML = '';
    compositeModelsContainer.innerHTML = '';
    Object.entries(configData.reasoner_models || {}).forEach(([name, config]) => { renderReasonerModel(name, config); });
    Object.entries(configData.target_models || {}).forEach(([name, config]) => { renderTargetModel(name, config); });
    Object.entries(configData.composite_models || {}).forEach(([name, config]) => { renderCompositeModel(name, config); });
}

function renderReasonerModel(name, config) {
    const template = document.getElementById('reasoner-model-template');
    const clone = document.importNode(template.content, true);
    clone.querySelector('.model-name').textContent = name;
    const form = clone.querySelector('.model-form');
    form.querySelector('.model-id').value = config.model_id || '';
    form.querySelector('.api-key').value = config.api_key || '';
    form.querySelector('.api-base-url').value = config.api_base_url || '';
    form.querySelector('.api-request-address').value = config.api_request_address || '';
    form.querySelector('.model-format').value = config.model_format || 'openai';
    form.querySelector('.is-origin-reasoning').checked = config.is_origin_reasoning || false;
    form.querySelector('.is-valid').checked = config.is_valid || false;
    form.querySelector('.is-proxy-open').checked = config.proxy_open || false;

    // 監聽格式變更以自動填入預設值
    form.querySelector('.model-format').addEventListener('change', (e) => {
        applyFormatPresets(form, e.target.value, 'reasoner');
    });

    form.querySelector('.save-model-btn').addEventListener('click', () => { saveReasonerModel(name, form); });
    clone.querySelector('.edit-model-btn').addEventListener('click', () => { toggleFormEditable(form, true); });
    clone.querySelector('.delete-model-btn').addEventListener('click', () => { showDeleteConfirmation('reasoner', name); });
    toggleFormEditable(form, false);
    reasonerModelsContainer.appendChild(clone);
}

function renderTargetModel(name, config) {
    const template = document.getElementById('target-model-template');
    const clone = document.importNode(template.content, true);
    clone.querySelector('.model-name').textContent = name;
    const form = clone.querySelector('.model-form');
    form.querySelector('.model-id').value = config.model_id || '';
    form.querySelector('.api-key').value = config.api_key || '';
    form.querySelector('.api-base-url').value = config.api_base_url || '';
    form.querySelector('.api-request-address').value = config.api_request_address || '';
    form.querySelector('.model-format').value = config.model_format || 'gemini';
    form.querySelector('.is-valid').checked = config.is_valid || false;
    form.querySelector('.is-proxy-open').checked = config.proxy_open || false;

    // 監聽格式變更以自動填入預設值
    form.querySelector('.model-format').addEventListener('change', (e) => {
        applyFormatPresets(form, e.target.value, 'target');
    });

    form.querySelector('.save-model-btn').addEventListener('click', () => { saveTargetModel(name, form); });
    clone.querySelector('.edit-model-btn').addEventListener('click', () => { toggleFormEditable(form, true); });
    clone.querySelector('.delete-model-btn').addEventListener('click', () => { showDeleteConfirmation('target', name); });
    toggleFormEditable(form, false);
    targetModelsContainer.appendChild(clone);
}

function renderCompositeModel(name, config) {
    const template = document.getElementById('composite-model-template');
    const clone = document.importNode(template.content, true);
    clone.querySelector('.model-name').textContent = name;
    const form = clone.querySelector('.model-form');
    form.querySelector('.model-id').value = config.model_id || '';
    form.querySelector('.is-valid').checked = config.is_valid || false;
    const reasonerSelect = form.querySelector('.reasoner-model-select');
    reasonerSelect.innerHTML = '';
    Object.keys(configData.reasoner_models || {}).forEach(modelName => {
        const option = document.createElement('option'); option.value = modelName; option.textContent = modelName; reasonerSelect.appendChild(option);
    });
    const targetSelect = form.querySelector('.target-model-select');
    targetSelect.innerHTML = '';
    Object.keys(configData.target_models || {}).forEach(modelName => {
        const option = document.createElement('option'); option.value = modelName; option.textContent = modelName; targetSelect.appendChild(option);
    });
    reasonerSelect.value = config.reasoner_models || '';
    targetSelect.value = config.target_models || '';
    form.querySelector('.save-model-btn').addEventListener('click', () => { saveCompositeModel(name, form); });
    clone.querySelector('.edit-model-btn').addEventListener('click', () => { toggleFormEditable(form, true); });
    clone.querySelector('.delete-model-btn').addEventListener('click', () => { showDeleteConfirmation('composite', name); });
    toggleFormEditable(form, false);
    compositeModelsContainer.appendChild(clone);
}

function renderProxySettings() {
    proxyOpenSwitch.checked = configData.proxy.proxy_open;
    proxyAddressInput.value = configData.proxy.proxy_address || '';
}

function renderSystemSettings() {
    const { allow_origins, log_level, api_key, save_deepseek_tokens, save_deepseek_tokens_max_tokens, max_reasoning_tokens } = configData.system;
    allowOriginsContainer.innerHTML = '';
    if (allow_origins && allow_origins.length > 0) { allow_origins.forEach(origin => { addAllowOriginInput(origin); }); }
    else { addAllowOriginInput('*'); }
    logLevelSelect.value = log_level || 'INFO';
    systemApiKeyInput.value = api_key || '123456';

    const saveDeepseekTokensSwitch = document.getElementById('save-deepseek-tokens');
    const deepseekTokensMaxInput = document.getElementById('deepseek-tokens-max');
    const deepseekTokensMaxContainer = document.getElementById('deepseek-tokens-max-container');
    const maxReasoningTokensInput = document.getElementById('max-reasoning-tokens');

    if (saveDeepseekTokensSwitch) {
        saveDeepseekTokensSwitch.checked = save_deepseek_tokens || false;
        if (deepseekTokensMaxContainer) { deepseekTokensMaxContainer.style.display = saveDeepseekTokensSwitch.checked ? 'block' : 'none'; }
        saveDeepseekTokensSwitch.addEventListener('change', function() {
            if (deepseekTokensMaxContainer) { deepseekTokensMaxContainer.style.display = this.checked ? 'block' : 'none'; }
        });
    }
    if (deepseekTokensMaxInput) { deepseekTokensMaxInput.value = save_deepseek_tokens_max_tokens || 5; }
    if (maxReasoningTokensInput) { maxReasoningTokensInput.value = max_reasoning_tokens || 0; }
}

function toggleFormEditable(form, editable) {
    form.querySelectorAll('input, select').forEach(input => { input.disabled = !editable; });
    form.querySelector('.save-model-btn').style.display = editable ? 'block' : 'none';
}

function applyFormatPresets(form, format, modelType) {
    // 格式預設值配置
    const presets = {
        openai: {
            reasoner: {
                base_url: 'https://api.deepseek.com',
                request_address: 'v1/chat/completions',
                is_origin_reasoning: true
            },
            target: {
                base_url: 'https://api.openai.com',
                request_address: 'v1/chat/completions'
            }
        },
        gemini: {
            reasoner: {
                base_url: 'https://generativelanguage.googleapis.com',
                request_address: '',
                is_origin_reasoning: false
            },
            target: {
                base_url: 'https://generativelanguage.googleapis.com',
                request_address: ''
            }
        },
        anthropic: {
            reasoner: {
                base_url: 'https://api.anthropic.com',
                request_address: 'v1/messages',
                is_origin_reasoning: false
            },
            target: {
                base_url: 'https://api.anthropic.com',
                request_address: 'v1/messages'
            }
        }
    };

    const preset = presets[format]?.[modelType];
    if (!preset) return;

    // 自動填入預設值
    if (preset.base_url) {
        form.querySelector('.api-base-url').value = preset.base_url;
    }
    if (preset.request_address !== undefined) {
        form.querySelector('.api-request-address').value = preset.request_address;
    }
    if (preset.is_origin_reasoning !== undefined && modelType === 'reasoner') {
        form.querySelector('.is-origin-reasoning').checked = preset.is_origin_reasoning;
    }

    showToast(`已載入 ${format} 格式預設值`, 'info');
}

function saveReasonerModel(name, form) {
    configData.reasoner_models[name] = {
        model_id: form.querySelector('.model-id').value, api_key: form.querySelector('.api-key').value,
        api_base_url: form.querySelector('.api-base-url').value, api_request_address: form.querySelector('.api-request-address').value,
        model_format: form.querySelector('.model-format').value,
        is_origin_reasoning: form.querySelector('.is-origin-reasoning').checked, is_valid: form.querySelector('.is-valid').checked,
        proxy_open: form.querySelector('.is-proxy-open').checked
    };
    toggleFormEditable(form, false);
    showToast(`推理模型 ${name} 已儲存`, 'success');
}

function saveTargetModel(name, form) {
    configData.target_models[name] = {
        model_id: form.querySelector('.model-id').value, api_key: form.querySelector('.api-key').value,
        api_base_url: form.querySelector('.api-base-url').value, api_request_address: form.querySelector('.api-request-address').value,
        model_format: form.querySelector('.model-format').value, is_valid: form.querySelector('.is-valid').checked,
        proxy_open: form.querySelector('.is-proxy-open').checked
    };
    toggleFormEditable(form, false);
    showToast(`目標模型 ${name} 已儲存`, 'success');
}

function saveCompositeModel(name, form) {
    configData.composite_models[name] = {
        model_id: form.querySelector('.model-id').value,
        reasoner_models: form.querySelector('.reasoner-model-select').value,
        target_models: form.querySelector('.target-model-select').value,
        is_valid: form.querySelector('.is-valid').checked
    };
    toggleFormEditable(form, false);
    showToast(`組合模型 ${name} 已儲存`, 'success');
}

function saveProxySettings() {
    try {
        configData.proxy.proxy_open = proxyOpenSwitch.checked;
        configData.proxy.proxy_address = proxyAddressInput.value.trim();
        saveAllConfigurations();
    } catch (error) { showToast('儲存代理設定失敗: ' + error.message, 'danger'); }
}

function saveSystemSettings() {
    try {
        const originInputs = document.querySelectorAll('.allow-origin');
        const origins = Array.from(originInputs).map(input => input.value.trim()).filter(value => value);
        const saveDeepseekTokensSwitch = document.getElementById('save-deepseek-tokens');
        const deepseekTokensMaxInput = document.getElementById('deepseek-tokens-max');
        const maxReasoningTokensInput = document.getElementById('max-reasoning-tokens');

        configData.system.allow_origins = origins;
        configData.system.log_level = logLevelSelect.value;
        configData.system.api_key = systemApiKeyInput.value.trim() || '123456';
        configData.system.save_deepseek_tokens = saveDeepseekTokensSwitch ? saveDeepseekTokensSwitch.checked : false;
        configData.system.save_deepseek_tokens_max_tokens = deepseekTokensMaxInput ? parseInt(deepseekTokensMaxInput.value) || 5 : 5;
        configData.system.max_reasoning_tokens = maxReasoningTokensInput ? parseInt(maxReasoningTokensInput.value) || 0 : 0;

        saveAllConfigurations();
    } catch (error) { showToast('儲存系統設定失敗: ' + error.message, 'danger'); }
}

async function saveAllConfigurations() {
    try {
        configData.proxy.proxy_open = proxyOpenSwitch.checked;
        configData.proxy.proxy_address = proxyAddressInput.value.trim();
        const originInputs = document.querySelectorAll('.allow-origin');
        configData.system.allow_origins = Array.from(originInputs).map(input => input.value.trim()).filter(value => value);
        configData.system.log_level = logLevelSelect.value;
        configData.system.api_key = systemApiKeyInput.value.trim() || '123456';
        const saveDeepseekTokensSwitch = document.getElementById('save-deepseek-tokens');
        const deepseekTokensMaxInput = document.getElementById('deepseek-tokens-max');
        const maxReasoningTokensInput = document.getElementById('max-reasoning-tokens');
        configData.system.save_deepseek_tokens = saveDeepseekTokensSwitch ? saveDeepseekTokensSwitch.checked : false;
        configData.system.save_deepseek_tokens_max_tokens = deepseekTokensMaxInput ? parseInt(deepseekTokensMaxInput.value) || 5 : 5;
        configData.system.max_reasoning_tokens = maxReasoningTokensInput ? parseInt(maxReasoningTokensInput.value) || 0 : 0;

        showToast('正在儲存設定...', 'info');
        const authApiKey = Auth.getCurrentApiKey();
        const response = await fetch(`${Auth.API_BASE_URL}/v1/config`, {
            method: 'POST', headers: { 'Authorization': `Bearer ${authApiKey}`, 'Content-Type': 'application/json' },
            body: JSON.stringify(configData)
        });
        if (!response.ok) { throw new Error('儲存設定失敗'); }
        showToast('所有設定已儲存', 'success');
    } catch (error) { showToast('儲存設定失敗: ' + error.message, 'danger'); }
}

function showAddModelModal(modelType) {
    let title;
    switch (modelType) {
        case 'reasoner': title = '新增推理模型'; break;
        case 'target': title = '新增目標模型'; break;
        case 'composite': title = '新增組合模型'; break;
    }
    addModelTitle.textContent = title;
    addModelForm.reset();
    addModelFields.innerHTML = '';
    addModelForm.dataset.modelType = modelType;
    addModelModal.show();
}

function handleAddModel() {
    const modelType = addModelForm.dataset.modelType;
    const modelName = document.getElementById('new-model-name').value.trim();
    if (!modelName) { showToast('請輸入模型名稱', 'warning'); return; }
    let targetCollection;
    switch (modelType) {
        case 'reasoner': targetCollection = configData.reasoner_models; break;
        case 'target': targetCollection = configData.target_models; break;
        case 'composite': targetCollection = configData.composite_models; break;
    }
    if (targetCollection[modelName]) { showToast(`模型 ${modelName} 已存在`, 'warning'); return; }
    let defaultConfig;
    switch (modelType) {
        case 'reasoner':
            defaultConfig = { model_id: '', api_key: '', api_base_url: '', api_request_address: '', model_format: 'openai', is_origin_reasoning: true, is_valid: true }; break;
        case 'target':
            defaultConfig = { model_id: '', api_key: '', api_base_url: '', api_request_address: '', model_format: 'gemini', is_valid: true }; break;
        case 'composite':
            const firstReasonerModel = Object.keys(configData.reasoner_models || {})[0] || '';
            const firstTargetModel = Object.keys(configData.target_models || {})[0] || '';
            defaultConfig = { model_id: modelName, reasoner_models: firstReasonerModel, target_models: firstTargetModel, is_valid: true }; break;
    }
    targetCollection[modelName] = defaultConfig;
    renderModels();
    addModelModal.hide();
    showToast(`模型 ${modelName} 已新增`, 'success');
}

function showDeleteConfirmation(modelType, modelName) {
    deleteModelNameSpan.textContent = modelName;
    confirmDeleteBtn.dataset.modelType = modelType;
    confirmDeleteBtn.dataset.modelName = modelName;
    confirmDeleteBtn.onclick = handleDeleteModel;
    confirmDeleteModal.show();
}

function handleDeleteModel() {
    const modelType = confirmDeleteBtn.dataset.modelType;
    const modelName = confirmDeleteBtn.dataset.modelName;
    switch (modelType) {
        case 'reasoner': delete configData.reasoner_models[modelName]; break;
        case 'target': delete configData.target_models[modelName]; break;
        case 'composite': delete configData.composite_models[modelName]; break;
    }
    renderModels();
    confirmDeleteModal.hide();
    showToast(`模型 ${modelName} 已刪除`, 'success');
}

function addAllowOriginInput(value = '') {
    const inputGroup = document.createElement('div'); inputGroup.className = 'input-group mb-2';
    const input = document.createElement('input'); input.type = 'text'; input.className = 'form-control allow-origin';
    input.placeholder = '例如: * 或 http://localhost:3000'; input.value = value;
    const button = document.createElement('button'); button.className = 'btn btn-outline-secondary remove-origin-btn'; button.type = 'button';
    button.innerHTML = '<i class="bi bi-trash"></i>';
    button.addEventListener('click', () => { if (document.querySelectorAll('.allow-origin').length > 1) { inputGroup.remove(); } });
    inputGroup.appendChild(input); inputGroup.appendChild(button);
    allowOriginsContainer.appendChild(inputGroup);
}

function showToast(message, type) {
    const toastContainer = document.getElementById('toast-container');
    const toastId = 'toast-' + Date.now();
    toastContainer.insertAdjacentHTML('beforeend', `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="3000">
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">提示</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">${message}</div>
        </div>`);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();
    toastElement.addEventListener('hidden.bs.toast', () => { toastElement.remove(); });
}

async function exportConfiguration() {
    try {
        showToast('正在匯出設定...', 'info');
        const apiKey = Auth.getCurrentApiKey();
        const response = await fetch(`${Auth.API_BASE_URL}/v1/config/export`, { headers: { 'Authorization': `Bearer ${apiKey}` } });
        if (!response.ok) { throw new Error('匯出設定失敗'); }
        const data = await response.json();
        const dataStr = JSON.stringify(data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const now = new Date();
        const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const link = document.createElement('a'); link.href = url; link.download = `deepclaude_config_${timestamp}.json`;
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
        URL.revokeObjectURL(url);
        showToast('設定匯出成功', 'success');
    } catch (error) { showToast('匯出設定失敗: ' + error.message, 'danger'); }
}

function handleConfigFileSelect(event) {
    const file = event.target.files[0];
    if (!file) { configPreview.classList.add('d-none'); confirmImportBtn.disabled = true; selectedConfigData = null; return; }
    if (!file.name.endsWith('.json')) { showToast('請選擇 JSON 格式的設定檔', 'warning'); configFileInput.value = ''; configPreview.classList.add('d-none'); confirmImportBtn.disabled = true; selectedConfigData = null; return; }
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const configContent = JSON.parse(e.target.result);
            selectedConfigData = configContent;
            displayConfigPreview(configContent);
            configPreview.classList.remove('d-none');
            confirmImportBtn.disabled = false;
        } catch (error) {
            showToast('設定檔格式不正確，請選擇有效的 JSON 檔案', 'danger');
            configFileInput.value = ''; configPreview.classList.add('d-none'); confirmImportBtn.disabled = true; selectedConfigData = null;
        }
    };
    reader.readAsText(file);
}

function displayConfigPreview(config) {
    const reasonerCount = Object.keys(config.reasoner_models || {}).length;
    const targetCount = Object.keys(config.target_models || {}).length;
    const compositeCount = Object.keys(config.composite_models || {}).length;
    const exportTime = config._export_metadata?.export_time || '未知';
    const exportSource = config._export_metadata?.source || '未知';
    configPreviewContent.innerHTML = `<div class="small">
        <div class="mb-2"><strong>設定統計：</strong></div>
        <ul class="mb-2"><li>推理模型：${reasonerCount} 個</li><li>目標模型：${targetCount} 個</li><li>組合模型：${compositeCount} 個</li></ul>
        <div class="mb-2"><strong>匯出資訊：</strong></div>
        <ul class="mb-0"><li>匯出時間：${exportTime}</li><li>來源：${exportSource}</li></ul>
    </div>`;
}

async function handleConfigImport() {
    if (!selectedConfigData) { showToast('請先選擇設定檔', 'warning'); return; }
    try {
        showToast('正在匯入設定...', 'info');
        const apiKey = Auth.getCurrentApiKey();
        const response = await fetch(`${Auth.API_BASE_URL}/v1/config/import`, {
            method: 'POST', headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
            body: JSON.stringify(selectedConfigData)
        });
        if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.error || '匯入設定失敗'); }
        importConfigModal.hide();
        configFileInput.value = ''; configPreview.classList.add('d-none'); confirmImportBtn.disabled = true; selectedConfigData = null;
        await loadConfigData();
        showToast('設定匯入成功', 'success');
    } catch (error) { showToast('匯入設定失敗: ' + error.message, 'danger'); }
}

window.Config = { init: initConfig, load: loadConfigData };
