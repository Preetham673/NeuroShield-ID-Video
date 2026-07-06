"""
verify.py

Face Verification using DeepFace (FaceNet)
"""

from deepface import DeepFace


def verify_faces(image1_path, image2_path):
    """
    Verify whether two face images belong to the same person.

    Parameters
    ----------
    image1_path : str
        Path to first image.

    image2_path : str
        Path to second image.

    Returns
    -------
    dict
        Verification results.
    """

    result = DeepFace.verify(
        img1_path=image1_path,
        img2_path=image2_path,
        model_name="Facenet",
        enforce_detection=False
    )

    similarity_score = max(0.0, (1 - result["distance"]) * 100)

    print("\n========== FACE VERIFICATION ==========\n")
    print(f"Verified        : {result['verified']}")
    print(f"Distance        : {result['distance']:.4f}")
    print(f"Threshold       : {result['threshold']:.4f}")
    print(f"Similarity Score: {similarity_score:.2f}%")

    return {
        "verified": result["verified"],
        "distance": result["distance"],
        "threshold": result["threshold"],
        "similarity_score": similarity_score,
    }