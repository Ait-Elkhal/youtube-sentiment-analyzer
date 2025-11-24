// chrome-extension/content.js

class YouTubeCommentExtractor {
    constructor() {
        this.commentSelectors = [
            '#content-text', // Nouveau design YouTube
            '#content',      // Ancien design
            '.ytd-comment-renderer #content-text',
            '.comment-text', // Fallback
            '[id*="comment"]' // Sélecteur générique
        ];
        this.observer = null;
        this.init();
    }

    init() {
        console.log('🎭 YouTube Sentiment Analyzer - Content Script chargé');
        
        // Écouter les messages du popup
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            if (request.action === 'extractComments') {
                this.extractComments(request.maxComments).then(comments => {
                    sendResponse({ success: true, comments });
                }).catch(error => {
                    sendResponse({ success: false, error: error.message });
                });
                return true; // Indique que la réponse sera asynchrone
            }
        });

        // Observer les changements de la page pour les chargements dynamiques
        this.setupMutationObserver();
    }

    setupMutationObserver() {
        this.observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.addedNodes.length > 0) {
                    // Des nouveaux commentaires ont peut-être été chargés
                    this.highlightCommentsIfNeeded();
                }
            });
        });

        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    async extractComments(maxComments = 50) {
        return new Promise((resolve, reject) => {
            try {
                // Attendre que la page soit complètement chargée
                if (document.readyState !== 'complete') {
                    window.addEventListener('load', () => this.doExtraction(maxComments, resolve, reject));
                } else {
                    this.doExtraction(maxComments, resolve, reject);
                }
            } catch (error) {
                reject(error);
            }
        });
    }

    doExtraction(maxComments, resolve, reject) {
        try {
            const comments = this.findComments();
            
            if (comments.length === 0) {
                reject(new Error('Aucun commentaire trouvé sur cette page'));
                return;
            }

            // Limiter le nombre de commentaires
            const limitedComments = comments.slice(0, maxComments);
            
            console.log(`📝 ${limitedComments.length} commentaires extraits`);
            resolve(limitedComments);
            
        } catch (error) {
            reject(error);
        }
    }

    findComments() {
        const comments = new Set();
        
        // Essayer différents sélecteurs
        for (const selector of this.commentSelectors) {
            const elements = document.querySelectorAll(selector);
            
            for (const element of elements) {
                const text = this.cleanCommentText(element.textContent);
                
                if (text && text.length > 5 && !this.isDuplicate(comments, text)) {
                    comments.add({
                        text: text,
                        element: element,
                        timestamp: new Date().toISOString()
                    });
                }
                
                // Arrêter si on a assez de commentaires
                if (comments.size >= 100) break;
            }
            
            if (comments.size > 0) break; // Arrêter si un sélecteur a fonctionné
        }

        return Array.from(comments);
    }

    cleanCommentText(text) {
        if (!text) return '';
        
        return text
            .trim()
            .replace(/\s+/g, ' ') // Remplacer les espaces multiples
            .replace(/[\r\n]+/g, ' ') // Remplacer les retours à la ligne
            .substring(0, 500); // Limiter la longueur
    }

    isDuplicate(commentsSet, text) {
        const normalizedText = text.toLowerCase().trim();
        for (const comment of commentsSet) {
            if (comment.text.toLowerCase().trim() === normalizedText) {
                return true;
            }
        }
        return false;
    }

    highlightCommentsIfNeeded() {
        // Cette fonction peut être utilisée pour surligner les commentaires
        // quand l'analyse est en cours (optionnel)
        const comments = this.findComments().slice(0, 10);
        
        comments.forEach(comment => {
            if (comment.element && !comment.element.classList.contains('sentiment-analyzed')) {
                comment.element.style.transition = 'background-color 0.3s ease';
            }
        });
    }

    // Méthode pour surligner les commentaires avec les résultats (optionnel)
    highlightCommentWithSentiment(element, sentiment, confidence) {
        const colors = {
            positive: 'rgba(52, 168, 83, 0.1)',
            neutral: 'rgba(251, 188, 5, 0.1)',
            negative: 'rgba(234, 67, 53, 0.1)'
        };
        
        const borderColors = {
            positive: '#34a853',
            neutral: '#fbbc05',
            negative: '#ea4335'
        };
        
        if (element && !element.classList.contains('sentiment-analyzed')) {
            element.style.backgroundColor = colors[sentiment];
            element.style.borderLeft = `3px solid ${borderColors[sentiment]}`;
            element.style.paddingLeft = '8px';
            element.classList.add('sentiment-analyzed');
            
            // Ajouter un badge de sentiment (optionnel)
            this.addSentimentBadge(element, sentiment, confidence);
        }
    }

    addSentimentBadge(element, sentiment, confidence) {
        const badge = document.createElement('span');
        badge.className = `sentiment-badge sentiment-${sentiment}`;
        badge.textContent = `${this.getSentimentEmoji(sentiment)} ${(confidence * 100).toFixed(0)}%`;
        badge.style.cssText = `
            position: absolute;
            top: 4px;
            right: 4px;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 10px;
            font-weight: bold;
            background: ${this.getBadgeColor(sentiment)};
            color: white;
            z-index: 1000;
        `;
        
        if (element.style.position !== 'relative') {
            element.style.position = 'relative';
        }
        
        element.appendChild(badge);
    }

    getSentimentEmoji(sentiment) {
        const emojis = {
            positive: '👍',
            neutral: '😐',
            negative: '👎'
        };
        return emojis[sentiment] || '❓';
    }

    getBadgeColor(sentiment) {
        const colors = {
            positive: '#34a853',
            neutral: '#fbbc05',
            negative: '#ea4335'
        };
        return colors[sentiment] || '#5f6368';
    }
}

// Initialiser l'extracteur quand le script est chargé
new YouTubeCommentExtractor();