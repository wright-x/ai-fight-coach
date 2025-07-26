# Email Setup Guide for AI Fight Coach

This guide will help you set up email functionality for user registration, analysis results, and feedback collection.

## Features

- **User Registration**: Welcome emails sent automatically when users sign up
- **Analysis Results**: Detailed analysis results emailed to users after video processing
- **Feedback Collection**: User feedback automatically sent to admin email (rudra@intunnelconsulting.com)

## Setup Instructions

### 1. Gmail Account Setup

#### Enable 2-Factor Authentication
1. Go to your Google Account settings: https://myaccount.google.com/
2. Navigate to Security
3. Enable 2-Step Verification if not already enabled

#### Generate App Password
1. In Google Account settings, go to Security
2. Click on "2-Step Verification"
3. Scroll down and click "App passwords"
4. Select "Mail" from the dropdown
5. Click "Generate"
6. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

### 2. Update Email Configuration

Edit the file `email_config.py` and update the SMTP configuration:

```python
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "your-actual-email@gmail.com",  # Your Gmail address
    "password": "your-16-char-app-password"  # The app password you generated
}
```

### 3. Test Email Functionality

1. Start the server: `python start_server.py`
2. Go to http://localhost:8000
3. Register a new user with a real email address
4. Check if the welcome email is received
5. Upload a video for analysis
6. Check if the analysis results email is received
7. Submit feedback
8. Check if the feedback email is sent to rudra@intunnelconsulting.com

## Email Templates

### Welcome Email
- Subject: "Welcome to AI Fight Coach! 🥊"
- Content: Welcome message with app features and next steps
- Sent: Immediately after user registration

### Analysis Results Email
- Subject: "Your AI Fight Coach Analysis Results - {name} 🥊"
- Content: Complete analysis results with highlights, drills, and YouTube recommendations
- Sent: After video analysis is completed

### Feedback Email (to Admin)
- Subject: "AI Fight Coach Feedback - {name}"
- Content: User rating, feedback text, and user information
- Sent: When user submits feedback
- Recipient: rudra@intunnelconsulting.com

## CSV User Tracking

All user registrations are automatically tracked in `users.csv` with the following columns:
- S.No: Sequential number
- Name: User's full name
- Email: User's email address
- Date of Sign Up: Timestamp of registration

## Troubleshooting

### Email Not Sending
1. Check if 2-Factor Authentication is enabled
2. Verify the app password is correct
3. Check Gmail account settings for any restrictions
4. Check server logs for SMTP errors

### Common Errors
- **Authentication failed**: Incorrect app password
- **Connection refused**: Check internet connection
- **Rate limit exceeded**: Gmail has daily sending limits

### Security Notes
- Never commit email credentials to version control
- Use app passwords instead of your main Gmail password
- Consider using environment variables for production

## Admin Access

To view all registered users, visit: http://localhost:8000/users

This endpoint returns all user data in JSON format for administrative purposes.

## Production Considerations

For production deployment:
1. Use environment variables for email credentials
2. Set up proper email service (SendGrid, AWS SES, etc.)
3. Implement email verification
4. Add rate limiting for email sending
5. Set up email templates in a proper template engine 