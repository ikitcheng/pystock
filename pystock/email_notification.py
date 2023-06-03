import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, username, password):
    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "plain"))

    # Convert the message to a string
    text = message.as_string()

    try:
        # Create a secure SSL/TLS connection with the SMTP server
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        # Login to the email account
        server.login(username, password)

        # Send the email
        server.sendmail(sender_email, receiver_email, text)
        print("Email notification sent successfully!")

    except Exception as e:
        print(f"Error sending email notification: {str(e)}")

    finally:
        # Close the SMTP server connection
        server.quit()

def main():

    # Usage
    subject = "Email Notification"
    body = "This is a sample email notification sent using Python."
    sender_email = "matthewkit@gmail.com"
    receiver_email = "matthewkit@gmail.com"
    smtp_server = "smtp.gmail.com"  # Gmail SMTP server
    smtp_port = 465  # SSL/TLS port
    username = os.environ.get('username', None) # Your Gmail email address
    password = os.environ.get('password', None)  # Your Gmail password or app-specific password

    send_email(subject, body, sender_email, receiver_email, smtp_server, smtp_port, username, password)

if __name__ == '__main__':
    main()
