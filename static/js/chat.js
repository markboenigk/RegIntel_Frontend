// Chat application JavaScript - SIMPLIFIED VERSION
class ChatApp {
    constructor() {
        this.conversationHistory = [];
        this.isLoading = false;
        this.lastSources = [];
        this.init();
    }

    init() {
        this.bindEvents();
        this.setInitialStatus();
        this.checkHealth();
        this.autoResizeTextarea();
        this.checkAuthStatus();
        this.loadLatestInfo();

        // Ensure initial collection state is set
        this.setInitialCollectionState();
    }

    bindEvents() {
        // Send message
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');

        if (sendButton && messageInput) {
            sendButton.addEventListener('click', () => this.sendMessage());
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // Clear chat
        const clearButton = document.getElementById('clearButton');
        if (clearButton) {
            clearButton.addEventListener('click', () => this.clearChat());
        }








        // Collection selector (header buttons)
        const collectionRssFeeds = document.getElementById('collectionRssFeeds');
        const collectionFdaWarningLetters = document.getElementById('collectionFdaWarningLetters');

        if (collectionRssFeeds) {
            collectionRssFeeds.addEventListener('click', () => this.switchCollection('rss_feeds'));
        }
        if (collectionFdaWarningLetters) {
            collectionFdaWarningLetters.addEventListener('click', () => this.switchCollection('fda_warning_letters'));
        }



        // Sources expand/collapse functionality
        const sourcesHeader = document.getElementById('sourcesHeader');
        const sourcesToggle = document.getElementById('sourcesToggle');
        const sourcesContent = document.getElementById('sourcesContent');

        if (sourcesHeader && sourcesToggle && sourcesContent) {
            sourcesHeader.addEventListener('click', () => {
                const isExpanded = sourcesContent.style.display !== 'none';
                sourcesContent.style.display = isExpanded ? 'none' : 'block';
                sourcesToggle.classList.toggle('expanded', !isExpanded);
            });
        }

        // Authentication events
        const logoutButton = document.getElementById('logoutButton');
        if (logoutButton) {
            logoutButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
        }

        // User dropdown menu - removed, replaced with simple buttons


    }

    autoResizeTextarea() {
        const textarea = document.getElementById('messageInput');
        if (textarea) {
            textarea.addEventListener('input', function () {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            });
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        if (!messageInput) return;

        const message = messageInput.value.trim();

        if (!message || this.isLoading) return;

        // Add user message to chat
        this.addMessageToChat('user', message);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get the currently selected collection from the active button
            const activeButton = document.querySelector('.collection-button.active');
            if (!activeButton) {
                console.error('❌ No active collection button found');
                return;
            }

            const selectedCollection = activeButton.getAttribute('data-collection');
            console.log('🔍 Selected collection:', selectedCollection);

            const response = await fetch(`/api/chat/${selectedCollection}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    conversation_history: this.conversationHistory
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Add assistant response to chat
                this.addMessageToChat('assistant', data.response);

                // Update sources
                this.updateSources(data.sources);

                // SIMPLIFIED: No reranking info to display
                // this.updateRerankingInfo(data.reranking_info);
            } else {
                this.addMessageToChat('assistant', `Error: ${data.detail || 'Failed to get response'}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessageToChat('assistant', 'Sorry, I encountered an error. Please try again.');
        } finally {
            this.hideTypingIndicator();
        }
    }

    addMessageToChat(role, content) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (role === 'user') {
            messageContent.innerHTML = `<i class="fas fa-user"></i>${this.escapeHtml(content)}`;
        } else if (role === 'assistant') {
            messageContent.innerHTML = `<i class="fas fa-robot"></i>${this.escapeHtml(content)}`;
        } else {
            messageContent.innerHTML = content;
        }

        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Add to conversation history
        this.conversationHistory.push({ role, content });

        // Keep only last 20 messages
        if (this.conversationHistory.length > 20) {
            this.conversationHistory = this.conversationHistory.slice(-20);
        }
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        // Get the current collection to show appropriate message
        const activeButton = document.querySelector('.collection-button.active');
        const collectionName = activeButton ? activeButton.getAttribute('data-collection') : 'rss_feeds';

        let message = 'Searching for relevant sources...';
        let icon = 'fas fa-brain';

        if (collectionName === 'fda_warning_letters') {
            message = 'Searching FDA warning letters...';
            icon = 'fas fa-exclamation-triangle';
        } else if (collectionName === 'rss_feeds') {
            message = 'Searching regulatory news...';
            icon = 'fas fa-newspaper';
        }

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';

        const typingContent = document.createElement('div');
        typingContent.className = 'typing-indicator';
        typingContent.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;

        typingDiv.appendChild(typingContent);
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        this.isLoading = true;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.isLoading = false;


    }

    updateSources(sources) {
        const sourcesList = document.getElementById('sourcesList');
        const sourcesCount = document.getElementById('sourcesCount');
        if (!sourcesList) return;

        // Limit sources to maximum 5
        const limitedSources = sources ? sources.slice(0, 5) : [];
        const totalSources = sources ? sources.length : 0;

        // Update the count indicator
        if (sourcesCount) {
            if (totalSources === 0) {
                sourcesCount.textContent = '(0)';
                sourcesCount.className = 'sources-count no-sources';
            } else {
                sourcesCount.textContent = `(${totalSources})`;
                sourcesCount.className = 'sources-count has-sources';
            }
        }

        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = '<p class="no-sources">No sources found for current query</p>';

            // Collapse sources section when there are no sources
            const sourcesContent = document.getElementById('sourcesContent');
            const sourcesToggle = document.getElementById('sourcesToggle');
            if (sourcesContent && sourcesToggle) {
                sourcesContent.style.display = 'none';
                sourcesToggle.classList.remove('expanded');
            }
            return;
        }

        sourcesList.innerHTML = '';
        limitedSources.forEach((source, index) => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-item';

            // Display confidence score if available
            const confidenceScore = source.score || source.confidence || this.calculateConfidenceScore(source, index, totalSources);
            const scoreText = confidenceScore ? `Confidence: ${(confidenceScore * 100).toFixed(1)}%` : '';

            // Create a more informative source display
            const metadata = source.metadata || {};
            let sourceTitle = '';
            let sourceDate = '';
            let sourceCompany = '';

            if (metadata.company_name && metadata.company_name !== 'Unknown Company') {
                sourceCompany = metadata.company_name;
            }
            if (metadata.letter_date && metadata.letter_date !== 'Unknown Date') {
                sourceDate = metadata.letter_date;
            }
            if (metadata.article_title && metadata.article_title !== 'Unknown Title') {
                sourceTitle = metadata.article_title;
            }

            // Show the most relevant metadata first
            let metadataDisplay = '';
            if (sourceCompany) {
                metadataDisplay += `<strong>Company:</strong> ${this.escapeHtml(sourceCompany)}`;
            }
            if (sourceDate) {
                metadataDisplay += metadataDisplay ? ` | <strong>Date:</strong> ${this.escapeHtml(sourceDate)}` : `<strong>Date:</strong> ${this.escapeHtml(sourceDate)}`;
            }
            if (sourceTitle) {
                metadataDisplay += metadataDisplay ? ` | <strong>Title:</strong> ${this.escapeHtml(sourceTitle)}` : `<strong>Title:</strong> ${this.escapeHtml(sourceTitle)}`;
            }

            sourceDiv.innerHTML = `
                <h4>Source ${index + 1} ${scoreText ? `<span class="confidence-badge">${scoreText}</span>` : ''}</h4>
                ${metadataDisplay ? `<p class="source-metadata">${metadataDisplay}</p>` : ''}
                <p class="source-text">${this.escapeHtml(source.text.substring(0, 200))}${source.text.length > 200 ? '...' : ''}</p>
            `;
            sourcesList.appendChild(sourceDiv);
        });

        // Show a message if sources were limited
        if (totalSources > 5) {
            const limitMessage = document.createElement('div');
            limitMessage.className = 'sources-limit-message';
            limitMessage.innerHTML = `<p><small><i class="fas fa-info-circle"></i> Showing top 5 sources out of ${totalSources} found</small></p>`;
            sourcesList.appendChild(limitMessage);
        }

        // Add confidence score explanation
        const confidenceInfo = document.createElement('div');
        confidenceInfo.className = 'confidence-info';
        confidenceInfo.innerHTML = `<p><small><i class="fas fa-info-circle"></i> Confidence scores indicate relevance to your query. Higher scores suggest better semantic matches.</small></p>`;
        sourcesList.appendChild(confidenceInfo);

        // Auto-expand sources section when there are sources
        if (totalSources > 0) {
            const sourcesContent = document.getElementById('sourcesContent');
            const sourcesToggle = document.getElementById('sourcesToggle');
            if (sourcesContent && sourcesToggle) {
                sourcesContent.style.display = 'block';
                sourcesToggle.classList.add('expanded');
            }
        }

        // Store the last sources for blinking restoration
        this.lastSources = sources;
    }

    calculateConfidenceScore(source, index, totalSources) {
        // Calculate a confidence score based on actual relevance and quality
        // This is a fallback when the backend doesn't provide scores

        if (totalSources === 0) return 0;

        let score = 0.0;

        // Get the current query and clean it
        const query = this.getCurrentQuery();
        if (!query || !source.text) {
            // Fallback to position-based scoring if no query or text
            return Math.max(0.8 - (index * 0.08), 0.4);
        }

        // Clean and normalize the query
        const queryLower = query.toLowerCase()
            .replace(/what are news about\s*:\s*/i, '')  // Remove "what are news about :"
            .replace(/what are news about/i, '')          // Remove "what are news about"
            .replace(/news about/i, '')                   // Remove "news about"
            .replace(/about/i, '')                        // Remove "about"
            .trim();

        if (!queryLower) {
            return Math.max(0.8 - (index * 0.08), 0.4);
        }

        // Extract the main entity/company name from the query
        const mainEntity = queryLower.split(/\s+/)[0]; // Take first word as main entity

        const textLower = source.text.toLowerCase();
        const titleLower = source.metadata?.article_title?.toLowerCase() || '';

        // CRITICAL: Company/Entity matching gets highest priority
        if (mainEntity && (textLower.includes(mainEntity) || titleLower.includes(mainEntity))) {
            score += 0.6; // Major bonus for matching the main entity
        }

        // Check for exact phrase matches in title (highest relevance)
        if (titleLower.includes(queryLower)) {
            score += 0.3;
        }

        // Check for exact phrase matches in text
        if (textLower.includes(queryLower)) {
            score += 0.2;
        }

        // Check for individual word matches (excluding common words)
        const queryWords = queryLower.split(/\s+/).filter(word =>
            word.length > 2 &&
            !['the', 'and', 'or', 'but', 'for', 'with', 'about', 'news', 'what', 'are'].includes(word)
        );

        let wordMatches = 0;
        queryWords.forEach(word => {
            if (textLower.includes(word) || titleLower.includes(word)) {
                wordMatches++;
            }
        });

        // Bonus for matching more query words
        if (wordMatches > 0) {
            score += (wordMatches / queryWords.length) * 0.15;
        }

        // Position penalty (much smaller now since relevance is more important)
        score -= (index * 0.05);

        // Quality bonuses (smaller impact)
        if (source.text) {
            const textLength = source.text.length;
            if (textLength >= 200 && textLength <= 800) {
                score += 0.02;
            }
        }

        // Metadata quality bonuses (smaller impact)
        if (source.metadata) {
            const metadata = source.metadata;
            if (metadata.article_title && metadata.article_title !== 'Unknown Title') {
                score += 0.01;
            }
        }

        // Ensure score is within bounds and add small random variation to break ties
        const finalScore = Math.min(Math.max(score, 0.1), 1.0);
        return finalScore;
    }

    getCurrentQuery() {
        // Try to get the current query from the last user message
        if (this.conversationHistory.length > 0) {
            const lastUserMessage = this.conversationHistory.findLast(msg => msg.role === 'user');
            return lastUserMessage ? lastUserMessage.content : '';
        }
        return '';
    }



    // SIMPLIFIED: Reranking info update function commented out
    // updateRerankingInfo(rerankingInfo) {
    //     // Update reranking information display if needed
    //     console.log('Reranking info:', rerankingInfo);
    // }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;

        chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <i class="fas fa-info-circle"></i>
                    Welcome! I'm RegIna,
                                your AI powered regulatory intelligence assistant. I can
                                help you with questions about FDA Warning Letters and the
                                recent regulatory news.
                </div>
            </div>
        `;
        this.conversationHistory = [];
        this.updateSources([]);

        // Reset sources count to 0
        const sourcesCount = document.getElementById('sourcesCount');
        if (sourcesCount) {
            sourcesCount.textContent = '(0)';
            sourcesCount.className = 'sources-count no-sources';
        }
    }







    setInitialStatus() {
        const apiStatusDot = document.getElementById('apiStatusDot');
        const milvusStatusDot = document.getElementById('milvusStatusDot');

        if (apiStatusDot) {
            apiStatusDot.className = 'status-dot checking';
            apiStatusDot.title = 'API: Checking...';
        }
        if (milvusStatusDot) {
            milvusStatusDot.className = 'status-dot checking';
            milvusStatusDot.title = 'Milvus: Checking...';
        }
    }

    setInitialCollectionState() {
        // Ensure RSS Feeds button is active by default
        const rssButton = document.getElementById('collectionRssFeeds');
        const fdaButton = document.getElementById('collectionFdaWarningLetters');

        if (rssButton && fdaButton) {
            rssButton.classList.add('active');
            fdaButton.classList.remove('active');
            console.log('✅ Initial collection state set: RSS Feeds active');
        } else {
            console.error('❌ Collection buttons not found');
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            const apiStatusDot = document.getElementById('apiStatusDot');
            const milvusStatusDot = document.getElementById('milvusStatusDot');

            if (response.ok) {
                if (apiStatusDot) {
                    apiStatusDot.className = 'status-dot healthy';
                    apiStatusDot.title = 'API: Healthy';
                }
                if (milvusStatusDot) {
                    milvusStatusDot.className = 'status-dot healthy';
                    milvusStatusDot.title = 'Milvus: Connected';
                }
            } else {
                if (apiStatusDot) {
                    apiStatusDot.className = 'status-dot error';
                    apiStatusDot.title = 'API: Error';
                }
                if (milvusStatusDot) {
                    milvusStatusDot.className = 'status-dot error';
                    milvusStatusDot.title = 'Milvus: Error';
                }
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const apiStatusDot = document.getElementById('apiStatusDot');
            const milvusStatusDot = document.getElementById('milvusStatusDot');

            if (apiStatusDot) {
                apiStatusDot.className = 'status-dot error';
                apiStatusDot.title = 'API: Error';
            }
            if (milvusStatusDot) {
                milvusStatusDot.className = 'status-dot error';
                milvusStatusDot.title = 'Milvus: Error';
            }
        }

        // RAG config check removed - no longer needed
    }





    async loadLatestInfo() {
        await Promise.all([
            this.loadLatestFDAWarningLetters(),
            this.loadLatestNews()
        ]);
    }

    async loadLatestFDAWarningLetters() {
        try {
            console.log('🔍 Loading latest FDA warning letters from Supabase...');
            console.log('🔍 Making request to /api/warning-letters/latest?limit=10');

            // Use the new Supabase endpoint
            const response = await fetch('/api/warning-letters/latest?limit=10');
            console.log('🔍 Response received:', response);
            console.log('🔍 Response status:', response.status);
            console.log('🔍 Response ok:', response.ok);

            const data = await response.json();
            console.log('🔍 Response data:', data);

            const tbody = document.getElementById('fdaWarningLettersBody');
            console.log('🔍 Found tbody element:', tbody);
            if (!tbody) {
                console.error('❌ No tbody element found for FDA warning letters');
                return;
            }

            if (data.success && data.warning_letters && data.warning_letters.length > 0) {
                console.log(`✅ Loaded ${data.warning_letters.length} warning letters from Supabase`);
                console.log('🔍 First warning letter:', data.warning_letters[0]);

                tbody.innerHTML = data.warning_letters.map(letter => `
                    <tr>
                        <td>${this.formatDate(letter.letter_date)}</td>
                        <td>${this.escapeHtml(letter.company_name)}</td>
                        <td>${this.escapeHtml(letter.subject)}</td>
                    </tr>
                `).join('');

                console.log('🔍 Updated tbody innerHTML');

                // Show success message
                this.showTableSuccessMessage('fdaWarningLettersTable', `Successfully loaded ${data.warning_letters.length} warning letters from Supabase`);
            } else {
                console.log('⚠️ No warning letters found or error occurred:', data.error || 'No data');
                console.log('🔍 Data structure:', data);
                tbody.innerHTML = '<tr><td colspan="3" class="no-data">No warning letters found</td></tr>';
            }
        } catch (error) {
            console.error('❌ Failed to load FDA warning letters from Supabase:', error);
            console.error('❌ Error details:', error.message, error.stack);
            const tbody = document.getElementById('fdaWarningLettersBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="3" class="error-message">
                            <div style="text-align: center; padding: 20px;">
                                <i class="fas fa-exclamation-triangle" style="color: #ff6b6b; font-size: 24px; margin-bottom: 10px;"></i>
                                <div>Failed to load warning letters from Supabase</div>
                                <div style="font-size: 0.8rem; margin-top: 5px; color: #86868b;">
                                    Error: ${error.message || 'Unknown error'}
                                </div>
                                <button onclick="window.location.reload()" style="margin-top: 10px; padding: 8px 16px; background: #ff6b6b; color: white; border: none; border-radius: 6px; cursor: pointer;">
                                    <i class="fas fa-redo"></i> Retry
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }
        }
    }

    async loadLatestNews() {
        try {
            const response = await fetch('/api/search?query=regulatory%20news&collection=rss_feeds&limit=5');
            const data = await response.json();

            const tbody = document.getElementById('recentNewsBody');
            if (!tbody) return;

            if (data.sources && data.sources.length > 0) {
                tbody.innerHTML = data.sources.map(source => `
                    <tr>
                        <td>${this.formatDate(source.metadata?.published_date || source.metadata?.date || 'N/A')}</td>
                        <td>${source.metadata?.source || source.metadata?.publisher || 'N/A'}</td>
                        <td>${this.truncateText(source.metadata?.title || source.content || 'N/A', 60)}</td>
                        <td><span class="status-badge news">Regulatory</span></td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = '<tr><td colspan="4" class="no-data">No news found</td></tr>';
            }
        } catch (error) {
            console.error('Failed to load latest news:', error);
            const tbody = document.getElementById('recentNewsBody');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="4" class="error-message">Failed to load data</td></tr>';
            }
        }
    }

    formatDate(dateString) {
        if (!dateString || dateString === 'N/A') return 'N/A';
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return 'N/A';
            return date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        } catch (error) {
            return 'N/A';
        }
    }

    truncateText(text, maxLength) {
        if (!text || text === 'N/A') return 'N/A';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }



    switchCollection(collectionName) {
        console.log(`🔄 Switching to collection: ${collectionName}`);

        // Clear the current chat when switching collections
        this.clearChat();

        // Update the collection display
        this.updateCollectionDisplay(collectionName);

        // Show a notification
        const collectionLabels = {
            'rss_feeds': 'Regulatory News',
            'fda_warning_letters': 'FDA Warning Letters'
        };

        this.addMessageToChat('system', `Switched to ${collectionLabels[collectionName]} collection. You can now ask questions about this data.`);

        // Reset sources count when switching collections
        const sourcesCount = document.getElementById('sourcesCount');
        if (sourcesCount) {
            sourcesCount.textContent = '(0)';
            sourcesCount.className = 'sources-count no-sources';
        }
    }

    updateCollectionDisplay(collectionName) {
        // Update any UI elements that show the current collection
        const collectionLabels = {
            'rss_feeds': '📰 Regulatory News',
            'fda_warning_letters': '⚠️ FDA Warning Letters'
        };

        // Update the collection button states
        const rssButton = document.getElementById('collectionRssFeeds');
        const fdaButton = document.getElementById('collectionFdaWarningLetters');

        if (rssButton) {
            rssButton.classList.toggle('active', collectionName === 'rss_feeds');
        }
        if (fdaButton) {
            fdaButton.classList.toggle('active', collectionName === 'fda_warning_letters');
        }

        console.log(`Current collection: ${collectionLabels[collectionName]}`);
    }

    showTableSuccessMessage(tableId, message) {
        // Show a temporary success message above the table
        const table = document.getElementById(tableId);
        if (!table) return;

        // Remove any existing success message
        const existingMessage = table.parentNode.querySelector('.table-success-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        // Create success message
        const successMessage = document.createElement('div');
        successMessage.className = 'table-success-message';
        successMessage.style.cssText = `
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.9rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 184, 148, 0.3);
            animation: slideInDown 0.5s ease-out;
        `;
        successMessage.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;

        // Insert before the table
        table.parentNode.insertBefore(successMessage, table);

        // Remove after 3 seconds
        setTimeout(() => {
            if (successMessage.parentNode) {
                successMessage.style.animation = 'slideOutUp 0.5s ease-in';
                setTimeout(() => successMessage.remove(), 500);
            }
        }, 3000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Authentication Methods
    async checkAuthStatus() {
        try {
            console.log('🔍 Checking authentication status...');

            // Debug: Check if we have any cookies
            console.log('🍪 All cookies:', document.cookie);

            const response = await fetch('/api/auth/status');
            console.log('📡 Auth status response:', response.status);

            if (response.ok) {
                const authData = await response.json();
                console.log('🔐 Auth data:', authData);

                if (authData.authenticated) {
                    console.log('✅ User is authenticated:', authData.user);
                    this.showAuthenticatedUser(authData.user);
                    this.enablePersonalFeatures();
                } else {
                    console.log('❌ User is not authenticated');
                    this.showUnauthenticatedUser();
                    this.disablePersonalFeatures();
                }
            } else {
                console.log('❌ Auth status request failed:', response.status);
                this.showUnauthenticatedUser();
                this.disablePersonalFeatures();
            }
        } catch (error) {
            console.log('❌ Auth status check error:', error);
            this.showUnauthenticatedUser();
            this.disablePersonalFeatures();
        }
    }

    showAuthenticatedUser(user) {
        console.log('👤 Showing authenticated user UI for:', user);

        const authSection = document.getElementById('authSection');
        const loginSection = document.getElementById('loginSection');
        const userName = document.getElementById('userName');

        console.log('🔍 Auth section element:', authSection);
        console.log('🔍 Login section element:', loginSection);
        console.log('🔍 User name element:', userName);

        if (authSection) {
            authSection.style.display = 'flex';
            console.log('✅ Auth section displayed');
        } else {
            console.log('❌ Auth section not found');
        }

        if (loginSection) {
            loginSection.style.display = 'none';
            console.log('✅ Login section hidden');
        } else {
            console.log('❌ Login section not found');
        }

        if (userName) {
            userName.textContent = user.full_name || user.email;
            console.log('✅ User name updated');
        } else {
            console.log('❌ User name element not found');
        }

        // Hide auth info message for authenticated users
        const authInfoMessage = document.getElementById('authInfoMessage');
        if (authInfoMessage) {
            authInfoMessage.style.display = 'none';
        }
    }

    showUnauthenticatedUser() {
        console.log('👤 Showing unauthenticated user UI');

        const authSection = document.getElementById('authSection');
        const loginSection = document.getElementById('loginSection');

        console.log('🔍 Auth section element:', authSection);
        console.log('🔍 Login section element:', loginSection);

        if (authSection) {
            authSection.style.display = 'none';
            console.log('✅ Auth section hidden');
        } else {
            console.log('❌ Auth section not found');
        }

        if (loginSection) {
            loginSection.style.display = 'flex';
            console.log('✅ Login section displayed');
        } else {
            console.log('❌ Login section not found');
        }

        // Show auth info message for unauthenticated users
        const authInfoMessage = document.getElementById('authInfoMessage');
        if (authInfoMessage) {
            authInfoMessage.style.display = 'block';
        }
    }

    async logout() {
        try {
            const response = await fetch('/auth/signout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                this.showUnauthenticatedUser();
                this.disablePersonalFeatures();
                // Redirect to home page
                window.location.href = '/';
            } else {
                console.error('Logout failed');
            }
        } catch (error) {
            console.error('Logout error:', error);
        }
    }

    enablePersonalFeatures() {
        console.log('🔓 Enabling personal features for authenticated user');
        // Add personal features like chat history, saved conversations, etc.
        // For now, just log that features are enabled
    }

    disablePersonalFeatures() {
        console.log('🔒 Disabling personal features for unauthenticated user');
        // Remove personal features - chat still works but no history saved
        // For now, just log that features are disabled
    }
}

// Initialize the chat application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const app = new ChatApp();

    // Debug: Check if we're coming back from login
    console.log('🚀 ChatApp initialized');
    console.log('🔍 Current URL:', window.location.href);
    console.log('🔍 Referrer:', document.referrer);

    // If we came from login page, wait a bit and recheck auth
    if (document.referrer.includes('/auth/login') || document.referrer.includes('/auth/signin')) {
        console.log('🔄 Coming from login page, waiting to recheck auth...');
        setTimeout(() => {
            console.log('🔄 Rechecking auth after login...');
            app.checkAuthStatus();
        }, 2000);
    }
});