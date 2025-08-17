#!/usr/bin/env python3
"""
Simple environment variable test script for CI debugging
"""

import os
from dotenv import load_dotenv

def main():
    print("🔧 Environment Variable Test Script")
    print("=" * 50)
    
    # Load .env file if it exists
    print("📁 Loading .env file...")
    load_dotenv()
    
    # Check all required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "MILVUS_URI", 
        "MILVUS_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_JWT_SECRET"
    ]
    
    print("\n🔍 Checking required environment variables:")
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first and last few characters for security
            display_value = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***"
            print(f"   {var}: ✅ Set ({display_value})")
        else:
            print(f"   {var}: ❌ Not set")
            missing_vars.append(var)
    
    # Check optional variables
    optional_vars = [
        "SUPABASE_SECRET_KEY",
        "FDA_WARNING_LETTERS_COLLECTION",
        "RSS_FEEDS_COLLECTION"
    ]
    
    print("\n🔍 Checking optional environment variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"   {var}: ✅ Set")
        else:
            print(f"   {var}: ⚠️ Not set (optional)")
    
    # Summary
    print("\n📊 Summary:")
    if missing_vars:
        print(f"❌ Missing required variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ All required environment variables are set")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 