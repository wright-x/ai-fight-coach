"""
Email Configuration for AI Fight Coach
Update these settings with your actual email credentials
"""

import os

# Gmail SMTP Configuration
# For Railway deployment, these will be set as environment variables
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": os.getenv("SMTP_EMAIL", "your-email@gmail.com"),  # From environment variable
    "password": os.getenv("SMTP_PASSWORD", "your-app-password")  # From environment variable
}

# Admin email for feedback
ADMIN_EMAIL = "rudra@intunnelconsulting.com"

# Email templates
WELCOME_EMAIL_SUBJECT = "Welcome to AI Fight Coach! 🥊"
ANALYSIS_EMAIL_SUBJECT = "Your AI Fight Coach Analysis Results - {name} 🥊"
FEEDBACK_EMAIL_SUBJECT = "AI Fight Coach Feedback - {name}"

# Instructions for setting up Gmail:
"""
To use Gmail for sending emails:

1. Enable 2-Factor Authentication on your Gmail account
2. Generate an App Password:
   - Go to Google Account settings
   - Security > 2-Step Verification > App passwords
   - Generate a new app password for "Mail"
   - Use this password in the SMTP_CONFIG above

3. For Railway deployment:
   - Set SMTP_EMAIL environment variable to your Gmail
   - Set SMTP_PASSWORD environment variable to your app password

4. Test the email functionality by registering a new user

Note: Never commit your actual email credentials to version control!
""" 