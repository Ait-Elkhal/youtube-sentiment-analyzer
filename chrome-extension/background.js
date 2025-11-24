// chrome-extension/background.js

class SentimentAnalyzerBackground {
    constructor() {
        this.init();
    }

    init() {
        console.log('🎭 YouTube Sentiment Analyzer - Background Script chargé');
        
        // Écouter l'installation de l'extension
        chrome.runtime.onInstalled.addListener(() => {
            this.onInstalled();
        });

        // Écouter les messages entre les scripts
        this.setupMessageHandling();
    }

    onInstalled() {
        console.log('🎭 YouTube Sentiment Analyzer installé');
        
        // Initialiser le stockage avec des valeurs par défaut
        chrome.storage.local.get(['apiUrl', 'maxComments', 'theme'], (result) => {
            if (!result.apiUrl) {
                chrome.storage.local.set({
                    apiUrl: 'https://hardyzona-youtube-sentiment-analyzer.hf.space', // ⚡ CHANGÉ
                    maxComments: 50,
                    theme: 'light'
                });
            }
        });
    }

    setupMessageHandling() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            console.log('Message reçu:', request);
            
            // Gérer les requêtes API depuis le popup
            if (request.action === 'apiRequest') {
                this.handleAPIRequest(request, sendResponse);
                return true; // Réponse asynchrone
            }
            
            // Relayer les messages entre popup et content script
            if (request.action === 'extractComments') {
                chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                    if (tabs[0]) {
                        chrome.tabs.sendMessage(tabs[0].id, request, (response) => {
                            sendResponse(response);
                        });
                    } else {
                        sendResponse({ success: false, error: 'Aucun onglet actif' });
                    }
                });
                return true; // Réponse asynchrone
            }
        });
    }

    async handleAPIRequest(request, sendResponse) {
        try {
            const settings = await this.getSettings();
            const response = await fetch(`${settings.apiUrl}${request.endpoint}`, {
                method: request.method || 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    ...request.headers
                },
                body: request.body ? JSON.stringify(request.body) : undefined
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            sendResponse({ success: true, data });
        } catch (error) {
            console.error('Erreur API:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    getSettings() {
        return new Promise((resolve) => {
            chrome.storage.local.get(['apiUrl', 'maxComments'], (result) => {
                resolve({
                    apiUrl: result.apiUrl || 'https://hardyzona-youtube-sentiment-analyzer.hf.space', // ⚡ CHANGÉ
                    maxComments: result.maxComments || 50
                });
            });
        });
    }
}

// Initialiser le background script
new SentimentAnalyzerBackground();