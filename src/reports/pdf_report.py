"""
pdf_report.py

Generate PDF Verification Report
"""

from pathlib import Path
from datetime import datetime
import uuid

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def generate_pdf(result):

    # Unique, timestamped name. The old fixed name overwrote the previous
    # report every time, so the Reports page could only ever show one file
    # and concurrent users clobbered each other's report.
    REPORT_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pdf_path = REPORT_DIR / f"verification_report_{stamp}_{uuid.uuid4().hex[:6]}.pdf"

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "NeuroShield-ID Verification Report",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now()}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(Paragraph("Deepfake Detection", styles["Heading2"]))

    story.append(
        Paragraph(
            f"Prediction: {result['deepfake']['prediction']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Confidence: {result['deepfake']['confidence']:.2f}%",
            styles["BodyText"]
        )
    )

    if result.get("is_video"):

        story.append(
            Paragraph(
                f"Input: Video ({result['deepfake'].get('total_frames_analyzed', 0)} frames analyzed)",
                styles["BodyText"]
            )
        )

        story.append(
            Paragraph(
                f"Average FAKE Probability: {result['deepfake'].get('avg_fake_probability', 0.0):.2f}%",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 15))

    story.append(Paragraph("Face Verification", styles["Heading2"]))

    story.append(
        Paragraph(
            f"Verified: {result['verification']['verified']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Similarity: {result['verification']['similarity_score']:.2f}%",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(Paragraph("Trust Score", styles["Heading2"]))

    story.append(
        Paragraph(
            f"Trust Score: {result['trust_score']:.2f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Status: {result['status']}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(Paragraph("Zero Knowledge Proof", styles["Heading2"]))

    story.append(
        Paragraph(
            f"Proof Generated: {result['zkp']['generated']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Proof Verified: {result['zkp']['verified']}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Policy Passed: {result['zkp']['policy_passed']}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(Paragraph("Blockchain", styles["Heading2"]))

    story.append(
        Paragraph(
            # proof_hash is None whenever nothing was stored on the chain
            # (policy failed). Paragraph(None) raised a TypeError, so the
            # report failed for every non-trusted verification.
            result["blockchain"]["proof_hash"] or "Not stored (policy not passed)",
            styles["Code"]
        )
    )

    pdf = SimpleDocTemplate(str(pdf_path))

    pdf.build(story)

    return pdf_path