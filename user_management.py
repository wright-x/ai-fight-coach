"""
User Management System for AI Fight Coach
Handles user registration, CSV tracking, email notifications, and feedback collection
"""

import csv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, csv_file: str = "users.csv", smtp_config: Dict = None):
        self.csv_file = csv_file
        self.smtp_config = smtp_config or {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email": "your-email@gmail.com",  # Update this
            "password": "your-app-password"   # Update this
        }
        self.admin_email = "rudra@intunnelconsulting.com"
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Ensure the CSV file exists with proper headers"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['S.No', 'Name', 'Email', 'Date of Sign Up'])
            logger.info(f"Created new CSV file: {self.csv_file}")
    
    def register_user(self, name: str, email: str) -> bool:
        """Register a new user and send welcome email"""
        try:
            # Get next serial number
            next_sno = self._get_next_serial_number()
            
            # Add user to CSV
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([next_sno, name, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
            logger.info(f"Registered user: {name} ({email})")
            
            # Send welcome email
            self._send_welcome_email(name, email)
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False
    
    def _get_next_serial_number(self) -> int:
        """Get the next serial number for new user"""
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                rows = list(reader)
                if not rows:
                    return 1
                return max(int(row[0]) for row in rows if row[0].isdigit()) + 1
        except Exception as e:
            logger.error(f"Error getting next serial number: {e}")
            return 1
    
    def _send_welcome_email(self, name: str, email: str):
        """Send welcome email to new user"""
        try:
            subject = "Welcome to AI Fight Coach! 🥊"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #667eea; margin-bottom: 10px;">🥊 Welcome to AI Fight Coach!</h1>
                        <p style="font-size: 18px; color: #666;">Hi {name},</p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px;">
                        <h2 style="margin-top: 0;">Your Account is Ready!</h2>
                        <p>Thank you for joining AI Fight Coach! You're now part of an exclusive community of passionate boxers who are taking their training to the next level.</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 10px; margin-bottom: 30px;">
                        <h3 style="color: #333; margin-top: 0;">What You Can Do Now:</h3>
                        <ul style="padding-left: 20px;">
                            <li>📹 Upload your boxing videos for AI analysis</li>
                            <li>🎯 Get personalized feedback on your technique</li>
                            <li>🏋️ Receive recommended drills and exercises</li>
                            <li>📺 Access curated YouTube training videos</li>
                            <li>🔄 Track your progress over time</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center; margin-bottom: 30px;">
                        <a href="https://www.ai-boxing.com/main" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block;">Start Your First Analysis</a>
                    </div>
                    
                    <div style="border-top: 2px solid #eee; padding-top: 20px; text-align: center; color: #666;">
                        <p>You're part of the special few that sees this in its early stages. Your feedback will help us make this tool even better for passionate boxers like you!</p>
                        <p style="font-size: 14px;">Best regards,<br>The AI Fight Coach Team</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self._send_email(email, subject, html_content)
            logger.info(f"Welcome email sent to {email}")
            
        except Exception as e:
            logger.error(f"Error sending welcome email: {e}")
    
    def send_analysis_results_email(self, name: str, email: str, analysis_result: Dict, video_url: str):
        """Send analysis results email to user"""
        try:
            subject = f"Your AI Fight Coach Analysis Results - {name} 🥊"
            
            # Format analysis results similar to website
            highlights_html = ""
            if analysis_result.get('highlights'):
                highlights_html = "<h3 style='font-size: 1.8rem; margin-bottom: 20px; color: #333; text-align: center;'>🎯 Key Highlights</h3>"
                for i, highlight in enumerate(analysis_result['highlights'], 1):
                    highlights_html += f"""
                    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="font-size: 1.3rem; margin-bottom: 15px; color: #667eea; font-weight: 600;">⏰ Highlight {i} ({highlight.get('timestamp', 0)}s)</h4>
                        <div style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px; color: #333;">📝 <b>Analysis:</b> <i>{highlight.get('detailed_feedback', '')}</i></div>
                        <div style="font-size: 1.1rem; color: #e74c3c; font-weight: 600;">⚡ <b>Action Required:</b> {highlight.get('action_required', '')}</div>
                    </div>
                    """
            
            drills_html = ""
            if analysis_result.get('recommended_drills'):
                drills_html = "<h3 style='font-size: 1.8rem; margin: 30px 0 20px 0; color: #333; text-align: center;'>🏋️ Recommended Drills</h3>"
                for drill in analysis_result['recommended_drills']:
                    drills_html += f"""
                    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="font-size: 1.3rem; margin-bottom: 15px; color: #28a745; font-weight: 600;">💪 {drill.get('drill_name', '')}</h4>
                        <div style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 15px; color: #333;">📋 <b>Description:</b> <i>{drill.get('description', '')}</i></div>
                        <div style="font-size: 1.1rem; color: #f39c12; font-weight: 600;">🎯 <b>Problem It Fixes:</b> {drill.get('problem_it_fixes', '')}</div>
                    </div>
                    """
            
            videos_html = ""
            if analysis_result.get('youtube_recommendations'):
                videos_html = "<h3 style='font-size: 1.8rem; margin: 30px 0 20px 0; color: #333; text-align: center;'>📺 YouTube Recommendations</h3>"
                for i, rec in enumerate(analysis_result['youtube_recommendations'], 1):
                    videos_html += f"""
                    <div style="margin-bottom: 25px; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h4 style="font-size: 1.3rem; margin-bottom: 15px; color: #007bff; font-weight: 600;">🔗 Video {i}: <a href="{rec.get('url', '')}" target="_blank" style="color:#007bff;font-weight:bold;text-decoration:underline;">{rec.get('title', '')}</a></h4>
                        <div style="font-size: 1.1rem; margin-bottom: 15px; color: #333;">🛠️ <b>Problem Solved:</b> <i>{rec.get('problem_solved', '')}</i></div>
                    </div>
                    """
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa;">
                <div style="max-width: 800px; margin: 0 auto; padding: 20px;">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h1 style="color: #667eea; margin-bottom: 10px;">🥊 Your AI Fight Coach Analysis</h1>
                        <p style="font-size: 18px; color: #666;">Hi {name}, your analysis is ready!</p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
                        <h2 style="margin-top: 0;">Analysis Complete!</h2>
                        <p>Your boxing video has been analyzed by our AI coach. Here are your personalized results and recommendations.</p>
                        <a href="{video_url}" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 25px; text-decoration: none; border-radius: 20px; font-weight: bold; display: inline-block; margin-top: 15px;">Watch Your Analysis Video</a>
                    </div>
                    
                    <div style="background: white; padding: 30px; border-radius: 15px; margin-bottom: 30px;">
                        {highlights_html}
                        {drills_html}
                        {videos_html}
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
                        <h3 style="margin-top: 0;">💬 Help Us Improve!</h3>
                        <p>You're part of the special few that sees this in its early stages. Your feedback will help us make this tool even better for passionate boxers like you!</p>
                        <a href="https://www.ai-boxing.com/main" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 25px; text-decoration: none; border-radius: 20px; font-weight: bold; display: inline-block; margin-top: 15px;">Give Us Feedback</a>
                    </div>
                    
                    <div style="text-align: center; color: #666; font-size: 14px;">
                        <p>Best regards,<br>The AI Fight Coach Team</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self._send_email(email, subject, html_content)
            logger.info(f"Analysis results email sent to {email}")
            
        except Exception as e:
            logger.error(f"Error sending analysis results email: {e}")
    
    def send_feedback_to_admin(self, user_name: str, user_email: str, rating: int, feedback_text: str):
        """Send feedback to admin email"""
        try:
            subject = f"AI Fight Coach Feedback - {user_name}"
            
            rating_emojis = {1: "😞", 2: "😐", 3: "😊", 4: "😍", 5: "🔥"}
            rating_text = {1: "Poor", 2: "Okay", 3: "Good", 4: "Great", 5: "Amazing"}
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #667eea;">📊 New User Feedback</h2>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin-top: 0;">User Information</h3>
                        <p><strong>Name:</strong> {user_name}</p>
                        <p><strong>Email:</strong> {user_email}</p>
                        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <h3 style="margin-top: 0;">Rating</h3>
                        <p style="font-size: 24px;">{rating_emojis.get(rating, "❓")} {rating_text.get(rating, "Unknown")} ({rating}/5)</p>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                        <h3 style="margin-top: 0;">Feedback</h3>
                        <p style="white-space: pre-wrap;">{feedback_text}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self._send_email(self.admin_email, subject, html_content)
            logger.info(f"Feedback sent to admin from {user_email}")
            
        except Exception as e:
            logger.error(f"Error sending feedback to admin: {e}")
    
    def _send_email(self, to_email: str, subject: str, html_content: str):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_config['email']
            msg['To'] = to_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(self.smtp_config['email'], self.smtp_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise
    
    def get_all_users(self) -> List[Dict]:
        """Get all registered users"""
        try:
            users = []
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    users.append(row)
            return users
        except Exception as e:
            logger.error(f"Error reading users: {e}")
            return [] 