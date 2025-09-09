# notifier.py
import os, smtplib, mimetypes
from email.message import EmailMessage
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads your .env

# --- Email settings ---
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")  # may be comma-separated

def send_email(subject: str, body: str, attachment: Path | None = None):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and EMAIL_FROM and EMAIL_TO):
        return False, "Email not configured"
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    if attachment and attachment.exists():
        ctype, _ = mimetypes.guess_type(str(attachment))
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(attachment, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=attachment.name)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    return True, "sent"

# --- Twilio (optional) ---
try:
    from twilio.rest import Client
except Exception:
    Client = None

TWILIO_SID   = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM  = os.getenv("TWILIO_FROM")
TWILIO_TO    = os.getenv("TWILIO_TO")

def send_sms(body: str):
    if not (Client and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and TWILIO_TO):
        return False, "SMS not configured"
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    client.messages.create(from_=TWILIO_FROM, to=TWILIO_TO, body=body)
    return True, "sent"

def notify_cat(camera_id: str, image_path: Path | None, confidence: float | None):
    subj = f"Cat spotted! ({camera_id})"
    body = f"Cat detected on {camera_id}. Conf={confidence}"
    # Try both; succeed if either works
    ok_email, _ = send_email(subj, body, image_path)
    ok_sms, _   = send_sms(f"{subj} Conf={confidence}")
    return ok_email or ok_sms
