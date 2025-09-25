import pandas as pd
from sqlalchemy import create_engine, text

# ----------------------------
# Config
# ----------------------------
SQL_URI = "mysql+pymysql://root:sql_my1country@localhost:3306/BTP"
NUM_DOCTORS = 10  # total doctors available

# ----------------------------
# Connect to DB
# ----------------------------
engine = create_engine(SQL_URI)

with engine.connect() as conn:
    # 1. Get the max PatientID currently in DoctorAllocation (or Patients table)
    max_alloc_pid = conn.execute(text("SELECT MAX(PatientID) FROM DoctorAllocation")).scalar() or 0
    max_patient_id = conn.execute(text("SELECT MAX(PatientID) FROM Patients")).scalar() or 0

    # 2. Determine new patient IDs that need allocation
    new_patient_ids = list(range(max_alloc_pid + 1, max_patient_id + 1))

    # 3. Allocate doctors in a round-robin fashion
    allocations = []
    for i, pid in enumerate(new_patient_ids):
        doctor_id = (i % NUM_DOCTORS) + 1  # cycle through 1..NUM_DOCTORS
        allocations.append({"PatientID": pid, "DoctorID": doctor_id})

    # 4. Convert to DataFrame
    df_alloc = pd.DataFrame(allocations)

    # 5. Append to DoctorAllocation table
    if not df_alloc.empty:
        df_alloc.to_sql("DoctorAllocation", con=engine, if_exists="append", index=False)
        print(f"Allocated {len(df_alloc)} new patients to doctors.")
    else:
        print("No new patients to allocate.")
