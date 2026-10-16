"""
verify.py

Face Verification using DeepFace (FaceNet)
"""

from deepface import DeepFace


def verify_faces(image1_path, image2_path, enforce_detection=True):
    """
    Verify whether two face images belong to the same person.

    Parameters
    ----------
    image1_path : str
        Path to first image.

    image2_path : str
        Path to second image.

    enforce_detection : bool
        If True (default) a face must be found in BOTH images, otherwise
        verification fails closed. With False, DeepFace embeds the whole
        picture when no face is found, so two identical non-face images
        (a wall, a blank frame) came back as VERIFIED with 100% similarity.
        Not acceptable for an identity-security system.

    Returns
    -------
    dict
        Verification results. On "no face" the result is not verified,
        similarity 0, and carries an "error" message.
    """

    try:

        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=image2_path,
            model_name="Facenet",
            enforce_detection=enforce_detection
        )

    except ValueError as e:

        # DeepFace raises ValueError when no face can be detected.
        print(f"\nFace verification failed: {e}")

        return {
            "verified": False,
            "distance": None,
            "threshold": None,
            "similarity_score": 0.0,
            "error": "No face detected in one or both images.",
        }

    similarity_score = max(0.0, (1 - result["distance"]) * 100)

    print("\n========== FACE VERIFICATION ==========\n")
    print(f"Verified        : {result['verified']}")
    print(f"Distance        : {result['distance']:.4f}")
    print(f"Threshold       : {result['threshold']:.4f}")
    print(f"Similarity Score: {similarity_score:.2f}%")

    return {
        "verified": bool(result["verified"]),
        "distance": result["distance"],
        "threshold": result["threshold"],
        "similarity_score": similarity_score,
    }