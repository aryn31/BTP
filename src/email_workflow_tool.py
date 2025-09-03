# src/tools/email_workflow_tool.py
from langchain.tools import tool
from src.email_workflow import run_email_workflow
import json

# @tool
# def escalate_via_email(patient_email: str, doctor_email: str) -> str:
#     """
#     Escalate a health alert by first contacting the patient. If no reply, escalate to the doctor.
#     Returns the final state summary.
#     """
#     result = run_email_workflow(patient_email, doctor_email)
#     return f"Workflow finished with status={result['status']}, " \
#            f"patient_response={result['patient_response']}, " \
#            f"doctor_response={result['doctor_response']}"

# @tool
# def escalate_via_email(patient_email: str, doctor_email: str) -> str:
#     """
#     Escalate a health alert by first contacting the patient. If no reply, escalate to the doctor.
#     Asks for manual confirmation before sending mails.
#     Returns the final state summary.
#     """
#     print("\n--- Escalation Workflow Requested ---")
#     print(f"Patient Email: {patient_email}")
#     print(f"Doctor Email: {doctor_email}")

#     confirm = input("Do you want to run the email escalation workflow? (yes/no): ").strip().lower()
#     if confirm != "yes":
#         print("✅ Email workflow skipped by user.")
#         return "Escalation aborted by user."

#     # Run actual workflow only if confirmed
#     result = run_email_workflow(patient_email, doctor_email)

#     return (
#         f"Workflow finished with status={result['status']}, "
#         f"patient_response={result['patient_response']}, "
#         f"doctor_response={result['doctor_response']}"
#     )

@tool
def escalate_via_email(json_input: str) -> str:
    """
    Escalate a health alert by first contacting the patient. If no reply, escalate to the doctor.
    Asks for manual confirmation before sending mails.
    
    Input:
        A JSON string with keys:
        - patient_email: str
        - doctor_email: str
    
    Returns:
        Final state summary of the workflow.
    """
    try:
        data = json.loads(json_input)
        patient_email = data["patient_email"]
        doctor_email = data["doctor_email"]

        print("\n--- Escalation Workflow Requested ---")
        print(f"Patient Email: {patient_email}")
        print(f"Doctor Email: {doctor_email}")

        confirm = input("Do you want to run the email escalation workflow? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("✅ Email workflow skipped by user.")
            return "Escalation aborted by user."

        # Run actual workflow only if confirmed
        result = run_email_workflow(patient_email, doctor_email)

        return (
            f"Workflow finished with status={result['status']}, "
            f"patient_response={result['patient_response']}, "
            f"doctor_response={result['doctor_response']}"
        )
    except Exception as e:
        return f"❌ Error in escalation: {str(e)}"
