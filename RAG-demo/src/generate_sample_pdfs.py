"""
Generates a small set of synthetic company-policy PDFs to use as the
knowledge base for the RAG demo. Each doc covers one topic with specific,
checkable facts (numbers, durations, thresholds) so retrieval quality can be
judged objectively rather than subjectively.

Swap this out for real PDFs by just dropping them into data/pdfs/ -- nothing
downstream cares how the PDFs got there.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

OUT_DIR = Path(__file__).parent.parent / "data" / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS = {
    "remote_work_policy.pdf": (
        "Remote Work Policy",
        [
            "Employees may work remotely up to 3 days per week, subject to manager approval. "
            "Fully remote arrangements require VP-level sign-off and are reviewed every 6 months.",

            "All remote employees must be reachable during core hours of 10:00 AM to 3:00 PM in "
            "their local timezone. Response time expectation for Slack messages during core hours is 30 minutes.",

            "The company provides a one-time home office stipend of $500 for remote employees, plus "
            "$50 per month for internet reimbursement. Stipend requests must be submitted within 90 days of hire.",

            "Remote employees are required to use company-issued VPN software when accessing internal "
            "systems. Personal devices are not permitted to access production databases under any circumstances.",
        ],
    ),
    "leave_policy.pdf": (
        "Leave and Time Off Policy",
        [
            "Full-time employees accrue 15 days of paid vacation per year during their first 2 years, "
            "increasing to 20 days after 2 years of tenure. Unused vacation days roll over up to a maximum of 5 days.",

            "Sick leave is capped at 10 days per calendar year and does not roll over. Employees must "
            "notify their manager before 9:00 AM on the day of absence for unplanned sick leave.",

            "Parental leave is 12 weeks fully paid for the primary caregiver and 6 weeks fully paid for "
            "the secondary caregiver, regardless of gender. Leave must be taken within 12 months of the birth or adoption date.",

            "Bereavement leave is 5 paid days for immediate family and 2 paid days for extended family. "
            "Requests for additional unpaid bereavement leave are handled on a case-by-case basis by HR.",
        ],
    ),
    "it_security_policy.pdf": (
        "IT Security Policy",
        [
            "All employee passwords must be at least 14 characters long and rotated every 90 days. "
            "Multi-factor authentication is mandatory for all systems that store customer data.",

            "Laptops must have full-disk encryption enabled before being issued. Any lost or stolen "
            "device must be reported to IT Security within 1 hour of discovery.",

            "Access to production databases requires a signed access request approved by the data owner "
            "and expires automatically after 30 days unless renewed.",

            "Phishing simulation tests are run quarterly. Employees who fail 2 consecutive simulations "
            "are required to complete a mandatory security training module within 5 business days.",
        ],
    ),
    "expense_policy.pdf": (
        "Travel and Expense Policy",
        [
            "Domestic flights must be booked in economy class unless the flight duration exceeds 6 hours, "
            "in which case premium economy is allowed with manager approval.",

            "Meal reimbursement is capped at $75 per day for domestic travel and $100 per day for "
            "international travel. Alcohol is not reimbursable under any circumstances.",

            "All expense reports must be submitted within 30 days of the expense being incurred. Reports "
            "submitted after 60 days will not be reimbursed except in extenuating circumstances approved by finance.",

            "Client entertainment expenses over $200 require pre-approval from a director-level employee "
            "or above, submitted at least 3 business days in advance.",
        ],
    ),
    "onboarding_policy.pdf": (
        "New Hire Onboarding Policy",
        [
            "New hires complete a 2-week onboarding program covering company systems, security training, "
            "and role-specific shadowing. IT equipment must be provisioned at least 3 business days before the start date.",

            "Every new hire is assigned a buddy from a different team for the first 30 days to help with "
            "informal questions and cultural onboarding, separate from their direct manager.",

            "The 90-day probationary period includes a formal check-in at day 30, day 60, and day 90, "
            "each documented in the HR system with manager and employee sign-off.",

            "New hires must complete mandatory compliance training, including data privacy and code of "
            "conduct modules, within the first 5 business days of employment.",
        ],
    ),
}


def build_pdf(filename: str, title: str, paragraphs: list[str]) -> None:
    path = OUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * inch)]
    for i, para in enumerate(paragraphs, 1):
        story.append(Paragraph(f"Section {i}", styles["Heading2"]))
        story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))
    doc.build(story)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print(f"Generating {len(DOCS)} sample PDFs into {OUT_DIR} ...")
    for fname, (title, paras) in DOCS.items():
        build_pdf(fname, title, paras)
    print("Done.")
