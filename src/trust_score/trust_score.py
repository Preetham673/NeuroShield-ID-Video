"""
trust_score.py

AI Trust Score Engine
for NeuroShield-ID
"""


from src.config import (
    TRUST_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
    DEEPFAKE_WEIGHT,
    FACE_WEIGHT,
)


def calculate_trust_score(
    deepfake_confidence,
    similarity_score
):
    """
    Calculate Trust Score.

    Parameters
    ----------
    deepfake_confidence : float
        Probability that the image is REAL (0-100). NOT the confidence of
        the predicted class; see _real_probability() in pipeline.py.

    similarity_score : float
        Face verification similarity (0-100)
    """

    trust_score = (
        DEEPFAKE_WEIGHT * deepfake_confidence +
        FACE_WEIGHT * similarity_score
    )

    if trust_score >= TRUST_THRESHOLD:

        status = "TRUSTED"

    elif trust_score >= SUSPICIOUS_THRESHOLD:

        status = "SUSPICIOUS"

    else:

        status = "REJECTED"

    return trust_score, status