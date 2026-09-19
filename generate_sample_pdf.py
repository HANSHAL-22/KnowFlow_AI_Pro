from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("sample_college_handbook.pdf", pagesize=A4)
styles = getSampleStyleSheet()
story = [
    Paragraph("KnowFlow AI Pro — Sample College Handbook", styles["Title"]),
    Spacer(1, 20),
    Paragraph("This is a fictional demo document for testing only.", styles["BodyText"]),
    Spacer(1, 18),
]

sections = [
    ("Academic Attendance",
     "Students should maintain at least 75 percent attendance in each course unless an approved institutional exception applies."),
    ("Examination Eligibility",
     "Students must satisfy the academic and attendance conditions published by the institution before appearing for examinations."),
    ("Placement Eligibility",
     "Placement eligibility may depend on academic performance, active backlogs, attendance, and the rules announced by the placement office."),
    ("Library Rules",
     "Students must return borrowed books by the due date. Overdue material may attract a fine according to library policy."),
    ("Hostel Guidelines",
     "Residents must follow published entry, exit, safety, and visitor rules.")
]

for title, body in sections:
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Paragraph(body, styles["BodyText"]))
    story.append(Spacer(1, 16))

doc.build(story)
print("Created sample_college_handbook.pdf")
