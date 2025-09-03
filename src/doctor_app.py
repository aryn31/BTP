import streamlit as st
import mysql.connector
from datetime import datetime

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sql_my1country",
        database="BTP"
    )

st.title("Doctor Portal")

# Select doctor
conn = get_db()
cur = conn.cursor()
cur.execute("SELECT DoctorID, DoctorName FROM Doctors")
doctors = cur.fetchall()
doctor_map = {name: did for did, name in doctors}
doctor_name = st.selectbox("Select Doctor", list(doctor_map.keys()))
doctor_id = doctor_map[doctor_name]

# Fetch escalations for this doctor
cur.execute("""
    SELECT a.AlertID, p.Name, ad.Message, ad.SentAt
    FROM AgentDoctorMessages ad
    JOIN Alerts a ON ad.AlertID = a.AlertID
    JOIN Patients p ON a.PatientID = p.PatientID
    WHERE ad.DoctorID=%s AND ad.MessageType='ESCALATION'
    ORDER BY ad.SentAt DESC
""", (doctor_id,))
alerts = cur.fetchall()

if alerts:
    alert_id, patient_name, msg, sent_at = alerts[0]  # latest escalation
    st.subheader(f"Escalation for Patient: {patient_name} (AlertID {alert_id})")
    st.write(f"**Agent (ESCALATION) [{sent_at}]**: {msg}")

    # Show doctor ↔ agent history
    cur.execute("""
        SELECT Sender, MessageType, Message, SentAt
        FROM AgentDoctorMessages
        WHERE AlertID=%s AND DoctorID=%s
        ORDER BY SentAt
    """, (alert_id, doctor_id))
    messages = cur.fetchall()

    for sender, mtype, msg, sent_at in messages:
        st.write(f"**{sender} ({mtype}) [{sent_at}]**: {msg}")

    # Doctor response box
    reply = st.text_area("Doctor's Response")
    if st.button("Send Response"):
        cur.execute("""
            INSERT INTO AgentDoctorMessages (AlertID, PatientID, DoctorID, Sender, MessageType, Message, SentAt)
            VALUES (%s,
                    (SELECT PatientID FROM Alerts WHERE AlertID=%s),
                    %s,
                    'Doctor',
                    'RESPONSE',
                    %s,
                    %s)
        """, (alert_id, alert_id, doctor_id, reply, datetime.now()))
        conn.commit()
        st.success("Response sent to Agent.")
        st.rerun()
else:
    st.info("No escalations yet.")

cur.close()
conn.close()
