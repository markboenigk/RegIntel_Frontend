# 🔒 Security Guide for RegIntel Frontend

## Environment Variables Setup

This application requires several environment variables to be configured securely. **Never commit your actual `.env` file to version control.**

### Required Environment Variables

1. **Copy the template:**
   ```bash
   cp env.template .env
   ```

2. **Fill in your actual values:**
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-actual-key-here
   
   # Milvus/Zilliz Configuration
   MILVUS_URI=https://your-instance.zilliz.com
   MILVUS_TOKEN=your-actual-token-here
   
   # Supabase Configuration
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-actual-anon-key-here
   SUPABASE_JWT_SECRET=your-actual-jwt-secret-here
   
   # Application Security
   SECRET_KEY=your-32-character-random-secret-here
   ```

### Security Best Practices

1. **Use strong, unique secrets** for each environment
2. **Rotate API keys regularly**
3. **Never share credentials** in code, logs, or documentation
4. **Use environment-specific configurations** for development/staging/production
5. **Monitor API usage** for unusual activity

### File Security

- ✅ `.env` - **NEVER commit** (automatically ignored)
- ✅ `env.template` - Safe to commit (template only)
- ✅ `requirements.txt` - Safe to commit
- ✅ Source code - Safe to commit (no hardcoded secrets)

### Deployment Security

- Use GitHub Secrets for CI/CD
- Use Vercel Environment Variables for production
- Use AWS Secrets Manager for enterprise deployments
- Enable 2FA on all service accounts

## Security Features

- **JWT Token Validation** with proper signature verification
- **Rate Limiting** to prevent abuse
- **Input Validation** to prevent injection attacks
- **CORS Protection** to restrict cross-origin requests
- **Secure Headers** to prevent common web vulnerabilities
- **Authentication Middleware** for protected routes

## Reporting Security Issues

If you discover a security vulnerability, please:
1. **DO NOT** create a public issue
2. **DO** email security@yourdomain.com
3. **DO** include detailed reproduction steps
4. **DO** wait for acknowledgment before disclosure

## Security Updates

This application is regularly updated for security patches. Always:
- Keep dependencies updated
- Monitor security advisories
- Test security changes in staging
- Deploy security updates promptly 