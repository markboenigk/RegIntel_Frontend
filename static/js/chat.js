// Chat application JavaScript - SIMPLIFIED VERSION
class ChatApp {
    constructor() {
        this.conversationHistory = [];
        this.isLoading = false;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadSettings();
        this.checkHealth();
        this.autoResizeTextarea();
    }

    bindEvents() {
        // Send message
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');

        sendButton.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Clear chat
        document.getElementById('clearButton').addEventListener('click', () => this.clearChat());

        // Add document modal
        document.getElementById('addDocumentButton').addEventListener('click', () => this.showDocumentModal());
        document.getElementById('cancelDocumentButton').addEventListener('click', () => this.hideDocumentModal());
        document.getElementById('submitDocumentButton').addEventListener('click', () => this.addDocument());
        document.querySelector('.close').addEventListener('click', () => this.hideDocumentModal());

        // Save API key
        document.getElementById('saveKeyButton').addEventListener('click', () => this.saveApiKey());

        // Collection selector
        document.getElementById('collectionSelect').addEventListener('change', (e) => this.switchCollection(e.target.value));

        // Test configuration
        document.getElementById('testConfigButton').addEventListener('click', () => this.testConfiguration());

        // Modal close on outside click
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('documentModal');
            if (e.target === modal) {
                this.hideDocumentModal();
            }
        });
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('messageInput');
        textarea.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isLoading) return;

        // Add user message to chat
        this.addMessageToChat('user', message);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get the currently selected collection
            const selectedCollection = document.getElementById('collectionSelect').value;

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
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';

        const typingContent = document.createElement('div');
        typingContent.className = 'typing-indicator';
        typingContent.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
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

        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = '<p class="no-sources">No sources found for current query</p>';
            return;
        }

        sourcesList.innerHTML = '';
        sources.forEach((source, index) => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-item';

            // SIMPLIFIED: No reranking scores to display
            const scoreText = ''; // source.score ? `Relevance: ${(source.score * 100).toFixed(1)}%` : '';

            sourceDiv.innerHTML = `
                <h4>Source ${index + 1}</h4>
                <p>${this.escapeHtml(source.text.substring(0, 200))}${source.text.length > 200 ? '...' : ''}</p>
                ${scoreText ? `<p class="score">${scoreText}</p>` : ''}
                ${source.metadata ? `<p><small>${this.escapeHtml(source.metadata)}</small></p>` : ''}
            `;
            sourcesList.appendChild(sourceDiv);
        });
    }

    // SIMPLIFIED: Reranking info update function commented out
    // updateRerankingInfo(rerankingInfo) {
    //     // Update reranking information display if needed
    //     console.log('Reranking info:', rerankingInfo);
    // }

    clearChat() {
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <i class="fas fa-info-circle"></i>
                    Welcome! I'm your AI assistant powered by ChatGPT and enhanced with RAG (Retrieval-Augmented Generation). 
                    I can help you with questions and provide context-aware responses using the knowledge base.
                </div>
            </div>
        `;
        this.conversationHistory = [];
        this.updateSources([]);
    }

    showDocumentModal() {
        document.getElementById('documentModal').style.display = 'block';
        document.getElementById('documentText').focus();
    }

    hideDocumentModal() {
        document.getElementById('documentModal').style.display = 'none';
        document.getElementById('documentText').value = '';
        document.getElementById('documentMetadata').value = '';
    }

    async addDocument() {
        const text = document.getElementById('documentText').value.trim();
        const metadata = document.getElementById('documentMetadata').value.trim();

        if (!text) {
            alert('Please enter document text');
            return;
        }

        const submitButton = document.getElementById('submitDocumentButton');
        const originalText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
        submitButton.disabled = true;

        try {
            const response = await fetch('/api/add-document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    metadata: metadata
                })
            });

            const data = await response.json();

            if (response.ok) {
                alert('Document added successfully!');
                this.hideDocumentModal();
            } else {
                alert(`Error: ${data.detail || 'Failed to add document'}`);
            }
        } catch (error) {
            console.error('Error adding document:', error);
            alert('Error adding document. Please try again.');
        } finally {
            submitButton.innerHTML = originalText;
            submitButton.disabled = false;
        }
    }

    saveApiKey() {
        const apiKey = document.getElementById('openaiKey').value.trim();
        if (apiKey) {
            localStorage.setItem('openai_api_key', apiKey);
            alert('API key saved!');
        } else {
            alert('Please enter an API key');
        }
    }

    loadSettings() {
        const savedApiKey = localStorage.getItem('openai_api_key');
        if (savedApiKey) {
            document.getElementById('openaiKey').value = savedApiKey;
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            const apiStatus = document.getElementById('apiStatus');
            const milvusStatus = document.getElementById('milvusStatus');
            if (response.ok) {
                apiStatus.textContent = '✅ Healthy';
                apiStatus.className = 'status-value healthy';
                milvusStatus.textContent = '✅ Connected';
                milvusStatus.className = 'status-value healthy';
            } else {
                apiStatus.textContent = '❌ Error';
                apiStatus.className = 'status-value error';
                milvusStatus.textContent = '❌ Unknown';
                milvusStatus.className = 'status-value error';
            }
        } catch (error) {
            console.error('Health check failed:', error);
            document.getElementById('apiStatus').textContent = '❌ Error';
            document.getElementById('apiStatus').className = 'status-value error';
            document.getElementById('milvusStatus').textContent = '❌ Unknown';
            document.getElementById('milvusStatus').className = 'status-value error';
        }

        // Check OpenAI key and RAG config
        await this.checkOpenAIKey();
        await this.checkRAGConfig();
    }

    async checkOpenAIKey() {
        try {
            const response = await fetch('/api/config');
            const data = await response.json();

            if (data.openai_configured) {
                document.getElementById('openaiKeyStatus').textContent = '✅ Configured';
                document.getElementById('openaiKeyStatus').className = 'status-value healthy';
            } else {
                document.getElementById('openaiKeyStatus').textContent = '❌ Not Configured';
                document.getElementById('openaiKeyStatus').className = 'status-value error';
            }
        } catch (error) {
            console.error('OpenAI key check failed:', error);
            document.getElementById('openaiKeyStatus').textContent = '❌ Error';
            document.getElementById('openaiKeyStatus').className = 'status-value error';
        }
    }

    async checkRAGConfig() {
        try {
            const response = await fetch('/api/config');
            const data = await response.json();

            const configText = `Strict: ${data.strict_rag_only ? 'Yes' : 'No'}, Rerank: ${data.enable_reranking ? 'Yes' : 'No'}`;
            document.getElementById('ragConfigStatus').textContent = configText;
            document.getElementById('ragConfigStatus').className = 'status-value healthy';
        } catch (error) {
            console.error('RAG config check failed:', error);
            document.getElementById('ragConfigStatus').textContent = '❌ Error';
            document.getElementById('ragConfigStatus').className = 'status-value error';
        }
    }

    async testConfiguration() {
        try {
            console.log('🧪 Testing configuration...');

            // Get the currently selected collection
            const selectedCollection = document.getElementById('collectionSelect').value;

            // Test search functionality
            const searchResponse = await fetch('/api/test-search?query=stryker&limit=3');
            const searchData = await searchResponse.json();

            console.log('🔍 Search test result:', searchData);

            if (searchData.sources_found > 0) {
                alert(`✅ Search working! Found ${searchData.sources_found} Stryker sources`);
            } else {
                alert(`❌ Search not working. Found ${searchData.sources_found} sources`);
            }

            // Test chat functionality with selected collection
            const chatResponse = await fetch(`/api/chat/${selectedCollection}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: 'What news do you have about Stryker?',
                    conversation_history: []
                })
            });

            const chatData = await chatResponse.json();
            console.log('💬 Chat test result:', chatData);

            if (chatData.sources && chatData.sources.length > 0) {
                alert(`✅ Chat working! Found ${chatData.sources.length} sources in ${selectedCollection}`);
            } else {
                alert(`❌ Chat not working. Found ${chatData.sources.length} sources in ${selectedCollection}`);
            }

        } catch (error) {
            console.error('Configuration test failed:', error);
            alert('❌ Configuration test failed: ' + error.message);
        }
    }

    switchCollection(collectionName) {
        console.log(`🔄 Switching to collection: ${collectionName}`);

        // Clear the current chat when switching collections
        this.clearChat();

        // Update the collection display
        this.updateCollectionDisplay(collectionName);

        // Show a notification
        const collectionLabels = {
            'rss_feeds': 'RSS Feeds (Regulatory Intelligence)',
            'fda_warning_letters': 'FDA Warning Letters'
        };

        this.addMessageToChat('system', `Switched to ${collectionLabels[collectionName]} collection. You can now ask questions about this data.`);
    }

        updateCollectionDisplay(collectionName) {
        // Update any UI elements that show the current collection
        const collectionLabels = {
            'rss_feeds': '📰 RSS Feeds Collection',
            'fda_warning_letters': '⚠️ FDA Warning Letters Collection'
        };
        
        // Update the collection indicator in the header
        const indicator = document.getElementById('collectionIndicator');
        if (indicator) {
            indicator.textContent = collectionLabels[collectionName];
        }
        
        console.log(`Current collection: ${collectionLabels[collectionName]}`);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chat application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
}); 