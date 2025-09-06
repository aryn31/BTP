import streamlit as st
from sqlalchemy import create_engine, text
import os
import pandas as pd

DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

st.title("👤 Patient Portal")

patient_id = st.number_input("Enter your Patient ID", min_value=1, step=1)

if patient_id:
    with engine.begin() as conn:
        # Get latest alert
        alert = conn.execute(
            text("SELECT * FROM Alerts WHERE PatientID=:pid ORDER BY CreatedAt DESC LIMIT 1"),
            {"pid": patient_id}
        ).fetchone()

    if alert:
        alert_id = alert.AlertID
        st.subheader(f"Latest Alert (ID: {alert_id})")

        # Show conversation history
        msgs = pd.read_sql(
            text("SELECT Sender, MessageType, Message, SentAt FROM AgentPatientMessages WHERE AlertID=:aid ORDER BY SentAt"),
            engine,
            params={"aid": alert_id}
        )
        st.table(msgs)

        # --- Check if patient already responded ---
        with engine.begin() as conn:
            already_responded = conn.execute(
                text("""
                    SELECT COUNT(*) FROM AgentPatientMessages 
                    WHERE AlertID=:aid AND Sender='Patient' AND MessageType='RESPONSE'
                """),
                {"aid": alert_id}
            ).scalar()

        if already_responded == 0:
            response = st.text_area("💬 Your Response")
            if st.button("Submit Response"):
                with engine.begin() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO AgentPatientMessages
                            (AlertID, PatientID, Sender, MessageType, Message)
                            VALUES (:aid, :pid, 'Patient', 'RESPONSE', :msg)
                        """),
                        {"aid": alert_id, "pid": patient_id, "msg": response}
                    )
                st.success("✅ Response submitted.")
        else:
            st.info("ℹ️ You have already responded to this alert.")
    else:
        st.info("No alerts found for this patient.")
