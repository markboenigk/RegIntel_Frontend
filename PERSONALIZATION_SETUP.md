# RegIntel Personalization Features Setup Guide

This guide explains how to set up and use the new personalization features that allow logged-in users to save and view their search query history.

## 🚀 What's New

- **Query History**: Automatically saves user queries when they're logged in
- **Personal Profile**: Users can view their recent queries in their profile
- **Query Management**: Users can delete individual queries or clear all history
- **Statistics**: Track total queries, collection usage, and activity

## 📋 Prerequisites

- Supabase project set up with authentication
- RegIntel Frontend application running
- User authentication system working

## 🗄️ Database Setup

### Step 1: Run the SQL Script

1. Go to your Supabase dashboard
2. Navigate to the SQL Editor
3. Copy and paste the contents of `database_setup.sql`
4. Click "Run" to execute the script

This will create:
- `user_queries` table for storing query history
- Proper indexes for performance
- Row Level Security (RLS) policies for data privacy
- Helper function for query statistics

### Step 2: Verify Table Creation

Check that the table was created successfully:
```sql
SELECT * FROM user_queries LIMIT 1;
```

## 🔧 How It Works

### For Unauthenticated Users
- ✅ Can use the chatbot normally
- ✅ Get the same AI responses and sources
- ❌ Queries are not saved
- ❌ No personalization features

### For Authenticated Users
- ✅ All chatbot functionality works
- ✅ Queries are automatically saved
- ✅ Can view query history in profile
- ✅ Can manage their query history
- ✅ See personalized statistics

## 📱 User Experience

### Chat Interface
- When a user sends a message, it's automatically saved (if logged in)
- No UI changes - works seamlessly in the background
- Users can continue chatting normally

### Profile Page
- New "Activity" section shows recent queries
- Each query displays:
  - The actual search text
  - Collection searched (RSS Feeds or FDA Warnings)
  - Timestamp
  - Response length and source count
  - Delete button for individual queries
- Buttons to refresh or clear all queries

## 🛡️ Security Features

- **Row Level Security (RLS)**: Users can only see their own queries
- **Authentication Required**: All query operations require valid user session
- **Data Isolation**: No user can access another user's query history
- **Secure Deletion**: Users can only delete their own queries

## 🔍 API Endpoints

### Save Query
```
POST /auth/queries
Body: {
  "query_text": "user's question",
  "collection_name": "rss_feeds",
  "response_length": 150,
  "sources_count": 3
}
```

### Get User Queries
```
GET /auth/queries?limit=20&offset=0
```

### Delete Specific Query
```
DELETE /auth/queries/{query_id}
```

### Clear All Queries
```
DELETE /auth/queries
```

## 🧪 Testing

### Test as Unauthenticated User
1. Open the app without logging in
2. Send a few messages to the chatbot
3. Verify responses work normally
4. Check that no queries are saved

### Test as Authenticated User
1. Log in to the app
2. Send messages to the chatbot
3. Go to your profile page
4. Navigate to the "Activity" section
5. Verify your queries appear
6. Test deleting individual queries
7. Test clearing all queries

## 🐛 Troubleshooting

### Queries Not Saving
- Check browser console for errors
- Verify user is properly authenticated
- Check Supabase RLS policies are active
- Ensure `user_queries` table exists

### Profile Page Errors
- Check authentication status
- Verify API endpoints are accessible
- Check browser console for network errors

### Database Issues
- Verify table structure matches the SQL script
- Check RLS policies are correctly applied
- Ensure proper permissions are granted

## 📊 Performance Considerations

- Queries are limited to 20 by default (configurable)
- Database indexes optimize query performance
- RLS policies ensure efficient data access
- Query saving happens asynchronously (doesn't block chat)

## 🔮 Future Enhancements

Potential features to add:
- Query analytics and insights
- Export query history
- Query categorization and tags
- Search within saved queries
- Query sharing between users
- Advanced filtering and sorting

## 📞 Support

If you encounter issues:
1. Check the browser console for errors
2. Verify database setup is correct
3. Test authentication flow
4. Check Supabase logs for database errors

## 🎯 Summary

The personalization features provide a seamless way for authenticated users to:
- Keep track of their search history
- Analyze their usage patterns
- Manage their personal data
- Get insights into their RAG interactions

All while maintaining the same excellent chatbot experience for both authenticated and unauthenticated users. 