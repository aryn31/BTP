import streamlit as st
import mysql.connector
from datetime import datetime

# DB connection
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sql_my1country",
        database="BTP"
    )

st.title("Patient Portal")

# Select patient (for demo, dropdown)
conn = get_db()
cur = conn.cursor()
cur.execute("SELECT PatientID, Name FROM Patients")
patients = cur.fetchall()
patient_map = {name: pid for pid, name in patients}
patient_name = st.selectbox("Select Patient", list(patient_map.keys()))
patient_id = patient_map[patient_name]

# Fetch active alerts for this patient
cur.execute("""
    SELECT AlertID, CreatedAt
    FROM Alerts
    WHERE PatientID=%s
    ORDER BY CreatedAt DESC
""", (patient_id,))
alerts = cur.fetchall()

if alerts:
    alert_id = alerts[0][0]  # latest alert
    st.subheader(f"Active Alert (ID {alert_id})")

    # Show conversation
    cur.execute("""
        SELECT Sender, MessageType, Message, SentAt
        FROM AgentPatientMessages
        WHERE AlertID=%s AND PatientID=%s
        ORDER BY SentAt
    """, (alert_id, patient_id))
    messages = cur.fetchall()

    for sender, mtype, msg, sent_at in messages:
        st.write(f"**{sender} ({mtype}) [{sent_at}]**: {msg}")

    # Patient response box
    reply = st.text_area("Your Response")
    if st.button("Send Response"):
        cur.execute("""
            INSERT INTO AgentPatientMessages (AlertID, PatientID, Sender, MessageType, Message, SentAt)
            VALUES (%s, %s, 'Patient', 'RESPONSE', %s, %s)
        """, (alert_id, patient_id, reply, datetime.now()))
        conn.commit()
        st.success("Response sent.")
        st.rerun()
else:
    st.info("No active alerts.")

cur.close()
conn.close()
