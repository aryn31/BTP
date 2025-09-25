import streamlit as st
from sqlalchemy import create_engine, text
import os
import pandas as pd

DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

st.title("👨‍⚕️ Doctor Portal")

doctor_id = st.number_input("Enter your Doctor ID", min_value=1, step=1)

if doctor_id:
    with engine.begin() as conn:
        # Get latest escalated alert
        alert = conn.execute(
            text("""
                SELECT a.AlertID, a.PatientID, p.Name
                FROM Alerts a
                JOIN Patients p ON a.PatientID=p.PatientID
                WHERE a.AlertID IN (
                    SELECT AlertID FROM AgentDoctorMessages WHERE DoctorID=:did
                )
                ORDER BY a.CreatedAt DESC LIMIT 1
            """),
            {"did": doctor_id}
        ).fetchone()

    if alert:
        alert_id = alert.AlertID
        st.subheader(f"Escalated Alert (ID: {alert_id}) for Patient {alert.Name}")

        # Show conversation history
        msgs = pd.read_sql(
            text("SELECT Sender, MessageType, Message, SentAt FROM AgentDoctorMessages WHERE AlertID=:aid ORDER BY SentAt"),
            engine,
            params={"aid": alert_id}
        )
        st.table(msgs)

        # --- Check if doctor already responded ---
        with engine.begin() as conn:
            already_responded = conn.execute(
                text("""
                    SELECT COUNT(*) FROM AgentDoctorMessages 
                    WHERE AlertID=:aid AND Sender='Doctor' AND MessageType='RESPONSE'
                """),
                {"aid": alert_id}
            ).scalar()

        if already_responded == 0:
            response = st.text_area("💬 Your Response")
            if st.button("Submit Response"):
                with engine.begin() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO AgentDoctorMessages
                            (AlertID, PatientID, DoctorID, Sender, MessageType, Message)
                            VALUES (:aid, :pid, :did, 'Doctor', 'RESPONSE', :msg)
                        """),
                        {"aid": alert_id, "pid": alert.PatientID, "did": doctor_id, "msg": response}
                    )
                print(f"🩺 Doctor {doctor_id} responded for Patient {alert.PatientID}, "f"Alert {alert_id}: {response}")
                st.success("✅ Response submitted.")
        else:
            st.info("ℹ️ You have already responded to this alert.")
    else:
        st.info("No escalated alerts found for this doctor.")
