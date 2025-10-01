# streamlit_family_portal.py
import streamlit as st
from sqlalchemy import create_engine, text
import os
import pandas as pd

# -----------------------------
# Config
# -----------------------------
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
engine = create_engine(DB_URI)

st.title("🏡 Family Member Portal")

# -----------------------------
# Fetch all patients with family member names
# -----------------------------
with engine.begin() as conn:
    patients_df = pd.read_sql(
        text("SELECT PatientID, Name AS PatientName, FamilyMemberName FROM Patients"),
        conn
    )

# Create a display string for selectbox
patients_df["display"] = patients_df.apply(lambda row: f"{row.PatientID} - {row.FamilyMemberName}", axis=1)

# -----------------------------
# Patient selection
# -----------------------------
selected = st.selectbox("Select Patient (ID - Family Member)", options=patients_df["display"])

if selected:
    # Extract patient_id from display string
    patient_id = int(selected.split(" - ")[0])

    with engine.begin() as conn:
        # Get latest escalated alert for this patient
        alert = conn.execute(
            text("""
                SELECT a.AlertID, a.PatientID, p.Name AS PatientName, p.FamilyMemberName
                FROM Alerts a
                JOIN Patients p ON a.PatientID = p.PatientID
                WHERE a.PatientID = :pid
                AND a.AlertID IN (
                    SELECT AlertID FROM AgentFamilyMessages WHERE MessageType='ESCALATION'
                )
                ORDER BY a.CreatedAt DESC
                LIMIT 1
            """),
            {"pid": patient_id}
        ).fetchone()

    if alert:
        alert_id = alert.AlertID
        st.subheader(f"Escalated Alert (ID: {alert_id})")
        st.write(f"**Patient:** {alert.PatientName}  |  **Family Member:** {alert.FamilyMemberName}")

        # Show conversation history
        msgs = pd.read_sql(
            text("""
                SELECT Sender, MessageType, Message, SentAt
                FROM AgentFamilyMessages
                WHERE AlertID=:aid
                ORDER BY SentAt
            """),
            engine,
            params={"aid": alert_id}
        )
        st.table(msgs)

        # --- Check if family member already responded ---
        already_responded = conn.execute(
            text("""
                SELECT COUNT(*) FROM AgentFamilyMessages 
                WHERE AlertID=:aid AND Sender='FamilyMember' AND MessageType='RESPONSE'
            """),
            {"aid": alert_id}
        ).scalar()

        if already_responded == 0:
            response = st.text_area("💬 Your Response")
            if st.button("Submit Response"):
                with engine.begin() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO AgentFamilyMessages
                            (AlertID, PatientID, Sender, MessageType, Message)
                            VALUES (:aid, :pid, 'FamilyMember', 'RESPONSE', :msg)
                        """),
                        {"aid": alert_id, "pid": patient_id, "msg": response}
                    )
                st.success("✅ Response submitted.")
        else:
            st.info("ℹ️ You have already responded to this alert.")
    else:
        st.info("No escalated alerts found for this patient.")
