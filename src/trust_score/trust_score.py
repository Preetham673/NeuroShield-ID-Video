"""
trust_score.py

AI Trust Score Engine
for NeuroShield-ID
"""


def calculate_trust_score(
    deepfake_confidence,
    similarity_score
):
    """
    Calculate Trust Score.

    Parameters
    ----------
    deepfake_confidence : float
        Confidence that image is REAL (0-100)

    similarity_score : float
        Face verification similarity (0-100)
    """

    trust_score = (
        0.6 * deepfake_confidence +
        0.4 * similarity_score
    )

    if trust_score >= 85:

        status = "TRUSTED"

    elif trust_score >= 60:

        status = "SUSPICIOUS"

    else:

        status = "REJECTED"

    return trust_score, status