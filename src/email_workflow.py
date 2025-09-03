# src/email_workflow.py
from typing import TypedDict, Literal, Optional
import time
from langgraph.graph import StateGraph, END

from .email_utils import send_email, check_inbox


# --------- Define state ---------
class WorkflowState(TypedDict):
    patient_email: str
    doctor_email: str
    status: Literal["INIT", "WAIT_PATIENT", "WAIT_DOCTOR", "DONE", "FAILED"]
    patient_response: Optional[str]
    doctor_response: Optional[str]


# --------- Nodes (steps) ---------
def notify_patient(state: WorkflowState):
    send_email(
        state["patient_email"],
        "High Glucose Alert",
        "Please confirm you took insulin/medication."
    )
    state["status"] = "WAIT_PATIENT"
    return state


def wait_for_patient(state: WorkflowState):
    for _ in range(12):  # wait 2 min (12x10s)
        time.sleep(10)
        reply = check_inbox(
            from_addr=state["patient_email"],
            subject_filter="Re: High Glucose Alert"
        )
        if reply:
            state["patient_response"] = reply
            state["status"] = "DONE"
            return state
    # no response
    state["status"] = "WAIT_DOCTOR"
    return state


def notify_doctor(state: WorkflowState):
    send_email(
        state["doctor_email"],
        "Patient Unresponsive",
        f"Patient {state['patient_email']} did not respond to alert."
    )
    return state


def wait_for_doctor(state: WorkflowState):
    for _ in range(18):  # wait 3 min (18x10s)
        time.sleep(10)
        reply = check_inbox(
            from_addr=state["doctor_email"],
            subject_filter="Re: Patient Unresponsive"
        )
        if reply:
            state["doctor_response"] = reply
            state["status"] = "DONE"
            return state
    state["status"] = "FAILED"
    return state


# --------- Build workflow graph ---------
workflow = StateGraph(WorkflowState)

workflow.add_node("notify_patient", notify_patient)
workflow.add_node("wait_for_patient", wait_for_patient)
workflow.add_node("notify_doctor", notify_doctor)
workflow.add_node("wait_for_doctor", wait_for_doctor)

workflow.set_entry_point("notify_patient")

workflow.add_edge("notify_patient", "wait_for_patient")
workflow.add_conditional_edges(
    "wait_for_patient",
    lambda s: "DONE" if s["status"] == "DONE" else "WAIT_DOCTOR",
    {"DONE": END, "WAIT_DOCTOR": "notify_doctor"}
)
workflow.add_edge("notify_doctor", "wait_for_doctor")
workflow.add_conditional_edges(
    "wait_for_doctor",
    lambda s: "DONE" if s["status"] == "DONE" else "FAILED",
    {"DONE": END, "FAILED": END}
)

app = workflow.compile()


# --------- Runner ---------
def run_email_workflow(patient_email, doctor_email):
    state = {
        "patient_email": patient_email,
        "doctor_email": doctor_email,
        "status": "INIT",
        "patient_response": None,
        "doctor_response": None,
    }
    final = app.invoke(state)
    return final
