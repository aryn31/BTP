import random
from faker import Faker
from sqlalchemy import create_engine, text
import os

# -----------------------------
# Config
# -----------------------------
DB_URI = os.getenv("DATABASE_URI", "mysql+pymysql://root:sql_my1country@localhost:3306/BTP")
fake = Faker()

engine = create_engine(DB_URI)

# -----------------------------
# Fetch patients and update family member info
# -----------------------------
with engine.begin() as conn:
    result = conn.execute(text("SELECT PatientID, Gender FROM Patients"))
    patients = result.fetchall()

    for pid, patient_gender in patients:
        patient_gender = (patient_gender or "").strip().lower()
        if patient_gender not in ["male", "female"]:
            continue  # skip invalid genders

        # Possible relations for all patients
        relations = ["Father", "Mother", "Brother", "Sister", "Spouse"]
        relation = random.choice(relations)

        # Generate name with gender matching relation (Spouse = opposite)
        if relation == "Father" or relation == "Brother":
            fam_name = fake.name_male()
        elif relation == "Mother" or relation == "Sister":
            fam_name = fake.name_female()
        elif relation == "Spouse":
            fam_name = fake.name_female() if patient_gender == "male" else fake.name_male()
        else:
            fam_name = fake.name()  # fallback

        # Only mobile number is Indian format
        fam_mobile = str(random.randint(6000000000, 9999999999))

        # Email derived from family member name
        first_name = fam_name.split()[0].lower()
        fam_email = f"{first_name}{random.randint(100,999)}@gmail.com"

        # Update the patient row
        conn.execute(
            text("""
                UPDATE Patients
                SET FamilyMemberName=:fam_name,
                    Relation=:relation,
                    FamilyMemberMobile=:fam_mobile,
                    FamilyMemberEmail=:fam_email
                WHERE PatientID=:pid
            """),
            {
                "fam_name": fam_name,
                "relation": relation,
                "fam_mobile": fam_mobile,
                "fam_email": fam_email,
                "pid": pid
            }
        )

print("✅ Family member info added; mobile in Indian format and email generated from name!")
