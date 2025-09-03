# src/email_utils.py
import smtplib, imaplib, email, time
from email.mime.text import MIMEText
from email.header import decode_header
import re

SMTP_SERVER = "smtp.gmail.com"
IMAP_SERVER = "imap.gmail.com"
EMAIL = "system@example.com"       # system account
PASSWORD = "your_app_password"     # Gmail App Password

def send_email(to, subject, body):
    msg = MIMEText(body)
    msg["From"] = EMAIL
    msg["To"] = to
    msg["Subject"] = subject

    with smtplib.SMTP_SSL(SMTP_SERVER, 465) as server:
        server.login(EMAIL, PASSWORD)
        server.sendmail(EMAIL, to, msg.as_string())
    print(f"📤 Sent email to {to}: {subject}")

# def check_inbox(from_addr=None, subject_filter=None):
#     with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
#         mail.login(EMAIL, PASSWORD)
#         mail.select("inbox")

#         status, data = mail.search(None, "ALL")
#         mail_ids = data[0].split()
#         for num in reversed(mail_ids[-10:]):  # check last 10
#             status, data = mail.fetch(num, "(RFC822)")
#             msg = email.message_from_bytes(data[0][1])
#             sender = msg["From"]
#             subject = msg["Subject"]
#             if (not from_addr or from_addr in sender) and (not subject_filter or subject_filter in subject):
#                 return msg.get_payload(decode=True).decode()
#     return None


def check_inbox(from_addr=None, subject_filter=None, lookback=10):
    try:
        with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
            mail.login(EMAIL, PASSWORD)
            mail.select("inbox")

            # Build IMAP search query
            criteria = ["ALL"]
            if from_addr:
                criteria.append(f'FROM "{from_addr}"')
            if subject_filter:
                criteria.append(f'SUBJECT "{subject_filter}"')

            status, data = mail.search(None, " ".join(criteria))
            if status != "OK" or not data[0]:
                return None  # no matching messages

            mail_ids = data[0].split()
            for num in reversed(mail_ids[-lookback:]):
                status, msg_data = mail.fetch(num, "(RFC822)")
                if status != "OK":
                    continue

                msg = email.message_from_bytes(msg_data[0][1])

                # Decode subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")

                sender = msg.get("From", "")

                # Extract body
                body = None
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        dispo = str(part.get("Content-Disposition", ""))
                        if "attachment" in dispo:
                            continue
                        if content_type == "text/plain":
                            body = part.get_payload(decode=True).decode(
                                part.get_content_charset() or "utf-8", errors="ignore"
                            )
                            break
                    if not body:  # fallback to HTML
                        for part in msg.walk():
                            if part.get_content_type() == "text/html":
                                html = part.get_payload(decode=True).decode(
                                    part.get_content_charset() or "utf-8", errors="ignore"
                                )
                                body = re.sub("<[^<]+?>", "", html)  # strip tags
                                break
                else:
                    body = msg.get_payload(decode=True).decode(
                        msg.get_content_charset() or "utf-8", errors="ignore"
                    )

                if body:
                    return {
                        "sender": sender.strip(),
                        "subject": subject.strip(),
                        "body": body.strip()
                    }

        return None
    except Exception as e:
        print(f"⚠️ Error in check_inbox: {e}")
        return None
