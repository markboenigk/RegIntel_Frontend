# 🔐 Environment Variables Template for RegIntel Frontend

**⚠️ IMPORTANT**: This is a template. Copy this file to `.env` and fill in your actual values.
**NEVER commit your actual `.env` file to version control!**

## 📋 **Required Environment Variables**

### **API Keys & Credentials**
```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Milvus Database
MILVUS_URI=your_milvus_uri_here
MILVUS_TOKEN=your_milvus_token_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
SUPABASE_JWT_SECRET=your_supabase_jwt_secret_here
```

### **Collection Names**
```bash
# Milvus Collections
FDA_WARNING_LETTERS_COLLECTION=fda_warning_letters
RSS_FEEDS_COLLECTION=rss_feeds
DEFAULT_COLLECTION=rss_feeds
```

### **Security Configuration**
```bash
# Security Settings
SECURITY_ENABLED=true
DEBUG_MODE=false  # Set to false in production

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,https://reg-intel.vercel.app
TRUSTED_HOSTS=localhost,127.0.0.1,reg-intel.vercel.app

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW=60
```

### **RAG Configuration**
```bash
# RAG Settings
STRICT_RAG_ONLY=true
ENABLE_RERANKING=false
RERANKING_MODEL=gpt-4
INITIAL_SEARCH_MULTIPLIER=3
```

### **Input Validation**
```bash
# Validation Limits
MAX_MESSAGE_LENGTH=2000
MAX_CONVERSATION_HISTORY=20
```

## 🚀 **Vercel-Specific Variables**

When deploying to Vercel, these are automatically set:
```bash
# Vercel automatically provides these
VERCEL=true
VERCEL_ENV=production  # or preview, development
VERCEL_URL=reg-intel.vercel.app
```

### **Custom Domain (if you have one)**
```bash
# Add your custom domain
CUSTOM_DOMAIN=yourdomain.com
```

## 🔒 **Security Best Practices**

1. **Never commit `.env` files** to version control
2. **Use strong, unique passwords** for each service
3. **Rotate API keys regularly**
4. **Use environment-specific values** (dev/staging/prod)
5. **Limit API key permissions** to minimum required

## 📁 **File Structure**

```
your-repo/
├── .env                    # ⚠️ NEVER COMMIT THIS FILE
├── .env.example           # ✅ Safe to commit (this template)
├── .gitignore             # ✅ Must include .env
├── index.py               # ✅ Safe to commit
└── ...
```

## 🚫 **What NOT to Commit**

- `.env` files
- API keys
- Database passwords
- JWT secrets
- Production URLs
- Personal information

## ✅ **What IS Safe to Commit**

- Code files
- Configuration templates
- Documentation
- README files
- Requirements files
- Docker files (without secrets)

## 🔧 **Setting Up Environment Variables**

### **Local Development**
1. Copy this template to `.env`
2. Fill in your actual values
3. Add `.env` to `.gitignore`

### **Vercel Deployment**
1. Go to your Vercel project dashboard
2. Navigate to Settings → Environment Variables
3. Add each variable from the template
4. Set appropriate values for production

### **Example .gitignore Entry**
```gitignore
# Environment variables
.env
.env.local
.env.production
.env.staging

# But allow templates
!.env.example
!.env.template
```

## 🧪 **Testing Your Configuration**

After setting up environment variables, test with:
```bash
# Check if all required variables are set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✅ Environment loaded successfully')"

# Verify security settings
python -c "from index import app; print('✅ App configured successfully')"
```

## 📞 **Need Help?**

If you encounter issues:
1. Check that all required variables are set
2. Verify variable names match exactly
3. Ensure no extra spaces or quotes
4. Test locally before deploying

---

**Remember**: Security first! Never expose sensitive information in public repositories. 