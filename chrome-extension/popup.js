// chrome-extension/popup.js

class SentimentAnalyzerPopup {
    constructor() {
        // ⚡ REMPLACÉ : URL locale par URL Hugging Face HardyZona
        this.apiUrl = 'https://hardyzona-youtube-sentiment-analyzer.hf.space';
        this.maxComments = 50;
        this.currentComments = [];
        this.currentTheme = 'light';
        
        this.initializeElements();
        this.loadSettings();
        this.checkAPIStatus();
        this.setupEventListeners();
    }

    initializeElements() {
        // Boutons
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.extractBtn = document.getElementById('extractBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.themeToggle = document.getElementById('themeToggle');
        this.saveSettings = document.getElementById('saveSettings');
        
        // Éléments de statut
        this.connectionStatus = document.getElementById('connectionStatus');
        
        // Sections
        this.statsSection = document.getElementById('statsSection');
        this.commentsSection = document.getElementById('commentsSection');
        
        // Éléments statistiques
        this.positiveCount = document.getElementById('positiveCount');
        this.positivePercent = document.getElementById('positivePercent');
        this.neutralCount = document.getElementById('neutralCount');
        this.neutralPercent = document.getElementById('neutralPercent');
        this.negativeCount = document.getElementById('negativeCount');
        this.negativePercent = document.getElementById('negativePercent');
        
        // Liste des commentaires
        this.commentsList = document.getElementById('commentsList');
        
        // Paramètres
        this.apiUrlInput = document.getElementById('apiUrl');
        this.maxCommentsInput = document.getElementById('maxComments');
    }

    setupEventListeners() {
        this.analyzeBtn.addEventListener('click', () => this.analyzeComments());
        this.extractBtn.addEventListener('click', () => this.extractComments());
        this.exportBtn.addEventListener('click', () => this.exportResults());
        this.copyBtn.addEventListener('click', () => this.copySummary());
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
        this.saveSettings.addEventListener('click', () => this.saveSettingsToStorage());
        
        // Filtres
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.filterComments(e.target.dataset.filter));
        });
    }

    async checkAPIStatus() {
        try {
            this.setStatus('analyzing', 'Vérification de l\'API HardyZona...');
            
            // ⚡ UTILISE l'URL HardyZona Hugging Face
            const response = await fetch(`${this.apiUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                // ⚡ ADAPTÉ : Message personnalisé HardyZona
                const statusMessage = data.model_loaded 
                    ? `API HardyZona Connectée ✅ - ${data.model_type || 'Modèle ML'}` 
                    : `API HardyZona - Modèle en cours de chargement...`;
                
                this.setStatus('online', statusMessage);
                this.analyzeBtn.disabled = false;
            } else {
                throw new Error('API non disponible');
            }
        } catch (error) {
            // ⚡ MESSAGE MIS À JOUR avec l'URL HardyZona
            this.setStatus('offline', 'API HardyZona Déconnectée - Vérifiez internet');
            this.analyzeBtn.disabled = true;
        }
    }

    setStatus(type, message) {
        this.connectionStatus.className = `status ${type}`;
        this.connectionStatus.querySelector('.status-text').textContent = message;
    }

    async extractComments() {
        try {
            this.setStatus('analyzing', 'Extraction des commentaires...');
            
            // Envoyer un message au content script pour extraire les commentaires
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'extractComments',
                maxComments: this.maxComments
            });
            
            if (response && response.success) {
                this.currentComments = response.comments;
                console.log('🔍 DEBUG Extraction réussie:', this.currentComments.length, 'commentaires');
                this.setStatus('online', `${response.comments.length} commentaires extraits`);
                this.updateCommentsList(this.currentComments);
                this.commentsSection.classList.remove('hidden');
            } else {
                throw new Error('Aucun commentaire trouvé');
            }
        } catch (error) {
            this.setStatus('offline', 'Erreur d\'extraction - Ouvrez une vidéo YouTube');
            console.error('Erreur extraction:', error);
        }
    }

    async analyzeComments() {
        if (this.currentComments.length === 0) {
            alert('Veuillez d\'abord extraire les commentaires');
            return;
        }
    
        try {
            this.setAnalyzingState(true);
            this.setStatus('analyzing', 'Analyse HardyZona en cours...');
    
            const texts = this.currentComments.map(comment => comment.text);
            console.log('🔍 DEBUG Textes à analyser:', texts.length, 'textes');
            console.log('🔍 DEBUG Premier texte:', texts[0]);
            
            // ⚡ UTILISE l'URL HardyZona Hugging Face
            const apiUrl = `${this.apiUrl}/predict/batch`;
            console.log('🔍 DEBUG Full URL:', apiUrl);
            
            const requestBody = {
                texts: texts.slice(0, this.maxComments)
            };
            console.log('🔍 DEBUG Request Body:', JSON.stringify(requestBody));
    
            console.log('🔍 DEBUG Avant fetch...');
            const response = await fetch(apiUrl, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
    
            console.log('🔍 DEBUG Après fetch - Status:', response.status);
            console.log('🔍 DEBUG Response OK:', response.ok);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('🔍 DEBUG Erreur HTTP détaillée:', {
                    status: response.status,
                    statusText: response.statusText,
                    errorText: errorText
                });
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }
    
            const data = await response.json();
            console.log('🔍 DEBUG Données API reçues:', data);
            
            // Vérifier la structure des données
            if (!data.predictions || !Array.isArray(data.predictions)) {
                console.error('🔍 DEBUG Structure invalide:', data);
                throw new Error('Format de réponse API invalide - predictions manquantes');
            }

            console.log('🔍 DEBUG Prédictions reçues:', data.predictions.length);
    
            // Fusionner les données
            this.currentComments = this.currentComments.map((comment, index) => {
                const prediction = data.predictions[index];
                if (prediction) {
                    return {
                        ...comment,
                        sentiment: prediction.sentiment,
                        confidence: prediction.confidence,
                        probabilities: prediction.probabilities
                    };
                }
                return comment;
            });
    
            console.log('🔍 DEBUG Commentaires mis à jour:', this.currentComments);
    
            // Mettre à jour l'interface
            if (data.statistics) {
                console.log('🔍 DEBUG Mise à jour statistiques:', data.statistics);
                this.updateStatistics(data.statistics);
                this.renderChart(data.statistics);
            }
            
            this.updateCommentsList(this.currentComments);
            
            // ⚡ MESSAGE MIS À JOUR avec signature HardyZona
            this.setStatus('online', `✅ Analyse HardyZona terminée - ${data.processing_time?.toFixed(2) || '0.00'}s`);
            this.exportBtn.disabled = false;
            this.copyBtn.disabled = false;
    
            console.log('🎉 Analyse HardyZona terminée avec succès');
    
        } catch (error) {
            console.error('🔍 DEBUG Erreur analyse complète:', error);
            console.error('🔍 DEBUG Stack trace:', error.stack);
            this.setStatus('offline', '❌ Erreur API HardyZona - Vérifiez la connexion');
            this.exportBtn.disabled = true;
            this.copyBtn.disabled = true;
        } finally {
            this.setAnalyzingState(false);
        }
    }

    setAnalyzingState(analyzing) {
        const btnText = this.analyzeBtn.querySelector('.btn-text');
        const spinner = this.analyzeBtn.querySelector('.loading-spinner');
        
        if (analyzing) {
            btnText.textContent = 'Analyse HardyZona...';
            spinner.classList.remove('hidden');
            this.analyzeBtn.disabled = true;
        } else {
            btnText.textContent = 'Analyser avec HardyZona';
            spinner.classList.add('hidden');
            this.analyzeBtn.disabled = false;
        }
    }

    updateStatistics(stats) {
        const distribution = stats.sentiment_distribution;
        
        this.positiveCount.textContent = distribution.positive.count;
        this.positivePercent.textContent = `${distribution.positive.percentage.toFixed(1)}%`;
        
        this.neutralCount.textContent = distribution.neutral.count;
        this.neutralPercent.textContent = `${distribution.neutral.percentage.toFixed(1)}%`;
        
        this.negativeCount.textContent = distribution.negative.count;
        this.negativePercent.textContent = `${distribution.negative.percentage.toFixed(1)}%`;
        
        this.statsSection.classList.remove('hidden');
    }

    updateCommentsList(comments, filter = 'all') {
        this.commentsList.innerHTML = '';
        
        const filteredComments = filter === 'all' 
            ? comments 
            : comments.filter(comment => comment.sentiment === filter);
        
        filteredComments.forEach(comment => {
            const commentElement = this.createCommentElement(comment);
            this.commentsList.appendChild(commentElement);
        });
    }

    createCommentElement(comment) {
        const div = document.createElement('div');
        div.className = 'comment-item';
        
        const sentimentEmoji = {
            'positive': '👍',
            'neutral': '😐',
            'negative': '👎'
        }[comment.sentiment];
        
        div.innerHTML = `
            <div class="comment-header">
                <span class="sentiment-badge ${comment.sentiment}">
                    ${sentimentEmoji} ${comment.sentiment}
                </span>
                <span class="confidence">${(comment.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="comment-text" title="${comment.text}">
                ${comment.text}
            </div>
            <div class="api-source">🤖 HardyZona AI</div>
        `;
        
        return div;
    }

    filterComments(filter) {
        // Mettre à jour les boutons de filtre
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        this.updateCommentsList(this.currentComments, filter);
    }

    renderChart(stats) {
        console.log('📊 Génération du graphique HardyZona...');
        const canvas = document.getElementById('sentimentChart');
        const ctx = canvas.getContext('2d');
        
        // Effacer le canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Créer un graphique à barres simple
        this.createBarChart(ctx, stats, canvas.width, canvas.height);
    }

    createBarChart(ctx, stats, width, height) {
        const distribution = stats.sentiment_distribution;
        const colors = {
            positive: '#34a853',
            neutral: '#fbbc05', 
            negative: '#ea4335'
        };
        
        const margin = 20;
        const chartWidth = width - margin * 2;
        const chartHeight = height - margin * 2 - 30;
        const barWidth = 40;
        const spacing = 20;
        
        // Trouver la valeur maximale pour l'échelle
        const maxPercentage = Math.max(
            distribution.positive.percentage,
            distribution.neutral.percentage, 
            distribution.negative.percentage
        );
        const scale = chartHeight / maxPercentage;
        
        const sentiments = [
            { key: 'positive', label: '👍', color: colors.positive },
            { key: 'neutral', label: '😐', color: colors.neutral },
            { key: 'negative', label: '👎', color: colors.negative }
        ];
        
        // Dessiner les barres
        sentiments.forEach((sentiment, index) => {
            const percentage = distribution[sentiment.key].percentage;
            const barHeight = (percentage / 100) * chartHeight * 0.8;
            const x = margin + index * (barWidth + spacing);
            const y = margin + chartHeight - barHeight;
            
            // Barre
            ctx.fillStyle = sentiment.color;
            ctx.fillRect(x, y, barWidth, barHeight);
            
            // Pourcentage au-dessus de la barre
            ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-primary');
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`${percentage.toFixed(1)}%`, x + barWidth/2, y - 5);
            
            // Émoji en bas
            ctx.fillText(sentiment.label, x + barWidth/2, margin + chartHeight + 20);
        });
        
        // Titre avec signature HardyZona
        ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--text-secondary');
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Analyse des Sentiments - HardyZona', width/2, 15);
    }

    exportResults() {
        if (this.currentComments.length === 0) {
            alert('Aucune donnée à exporter');
            return;
        }

        const data = {
            analyzedAt: new Date().toISOString(),
            totalComments: this.currentComments.length,
            statistics: this.getCurrentStatistics(),
            comments: this.currentComments,
            // ⚡ AJOUTÉ : Information sur l'API HardyZona
            apiSource: 'HardyZona YouTube Sentiment API',
            apiUrl: 'https://hardyzona-youtube-sentiment-analyzer.hf.space',
            author: 'HardyZona - INDIA ENSAM Rabat'
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `youtube-sentiment-hardyzona-${new Date().getTime()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    copySummary() {
        if (this.currentComments.length === 0) {
            alert('Aucun résumé à copier');
            return;
        }

        const stats = this.getCurrentStatistics();
        const summary = `
🎭 Analyse des Sentiments YouTube - HardyZona AI

📊 Statistiques:
• 👍 Positifs: ${stats.positive.count} (${stats.positive.percentage.toFixed(1)}%)
• 😐 Neutres: ${stats.neutral.count} (${stats.neutral.percentage.toFixed(1)}%)
• 👎 Négatifs: ${stats.negative.count} (${stats.negative.percentage.toFixed(1)}%)

📈 Sentiment dominant: ${stats.dominant.sentiment}
⏱️ Analysé le: ${new Date().toLocaleString()}

🚀 Powered by HardyZona Sentiment Analysis
🌐 API: https://hardyzona-youtube-sentiment-analyzer.hf.space
        `.trim();

        navigator.clipboard.writeText(summary).then(() => {
            alert('Résumé HardyZona copié dans le presse-papier!');
        });
    }

    getCurrentStatistics() {
        const sentiments = this.currentComments.map(c => c.sentiment);
        const positive = sentiments.filter(s => s === 'positive').length;
        const neutral = sentiments.filter(s => s === 'neutral').length;
        const negative = sentiments.filter(s => s === 'negative').length;
        const total = this.currentComments.length;

        const stats = {
            positive: { count: positive, percentage: (positive / total) * 100 },
            neutral: { count: neutral, percentage: (neutral / total) * 100 },
            negative: { count: negative, percentage: (negative / total) * 100 }
        };

        // Trouver le sentiment dominant
        const dominant = Object.entries(stats).reduce((max, [sentiment, data]) => 
            data.count > max.count ? { sentiment, ...data } : max
        , { sentiment: 'neutral', count: 0 });

        return { ...stats, dominant };
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        this.themeToggle.textContent = this.currentTheme === 'light' ? '🌙' : '☀️';
        this.saveToStorage('theme', this.currentTheme);
        
        // Re-rendre le chart avec les nouvelles couleurs
        if (this.currentComments.length > 0) {
            const stats = this.getCurrentStatistics();
            this.renderChart(stats);
        }
    }

    loadSettings() {
        chrome.storage.local.get(['apiUrl', 'maxComments', 'theme'], (result) => {
            // ⚡ PAR DÉFAUT : URL HardyZona Hugging Face
            this.apiUrl = result.apiUrl || 'https://hardyzona-youtube-sentiment-analyzer.hf.space';
            this.maxComments = result.maxComments || 50;
            this.currentTheme = result.theme || 'light';
            
            this.apiUrlInput.value = this.apiUrl;
            this.maxCommentsInput.value = this.maxComments;
            
            document.documentElement.setAttribute('data-theme', this.currentTheme);
            this.themeToggle.textContent = this.currentTheme === 'light' ? '🌙' : '☀️';
        });
    }

    saveSettingsToStorage() {
        this.apiUrl = this.apiUrlInput.value.trim();
        this.maxComments = parseInt(this.maxCommentsInput.value);
        
        chrome.storage.local.set({
            apiUrl: this.apiUrl,
            maxComments: this.maxComments
        }, () => {
            alert('Paramètres HardyZona sauvegardés!');
            this.checkAPIStatus(); // Re-vérifier l'API avec la nouvelle URL
        });
    }

    saveToStorage(key, value) {
        chrome.storage.local.set({ [key]: value });
    }
}

// Initialiser l'application quand le DOM est chargé
document.addEventListener('DOMContentLoaded', () => {
    new SentimentAnalyzerPopup();
});