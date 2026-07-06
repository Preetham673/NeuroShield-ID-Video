"""
pipeline.py

NeuroShield-ID Integrated Pipeline
"""

from src.deepfake_detection.inference import predict
from src.face_verification.verify import verify_faces
from src.trust_score.trust_score import calculate_trust_score
from src.zkp.generate_proof import generate_proof
from src.zkp.verify_proof import verify_proof
from src.blockchain.store_proof import store_proof
from src.explainability.gradcam import generate_gradcam
from datetime import datetime
from src.database.database import save_verification

def run_pipeline(reference_image, verification_image):
    """
    Run the complete NeuroShield-ID pipeline.
    """

    print("\n===================================")
    print("        NeuroShield-ID")
    print("===================================")

    # ---------------------------------------
    # Module 1 - Deepfake Detection
    # ---------------------------------------

    deepfake_result = predict(verification_image)

    # ---------------------------------------
    # Module 2 - Face Verification
    # ---------------------------------------

    verification_result = verify_faces(
        reference_image,
        verification_image
    )

    # ---------------------------------------
    # Module 3 - Trust Score
    # ---------------------------------------

    trust_score, status = calculate_trust_score(
        deepfake_result["confidence"],
        verification_result["similarity_score"]
    )

    # ---------------------------------------
    # Module 4 - Zero Knowledge Proof
    # ---------------------------------------

    proof_generated = generate_proof(trust_score)

    zkp_result = {
    "proof_verified": False,
    "policy_passed": False
}

    if proof_generated:
        zkp_result = verify_proof()


    proof_hash = None

    if zkp_result["proof_verified"]:
     proof_hash = store_proof() 

    save_verification(

    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

    prediction=deepfake_result["prediction"],

    confidence=deepfake_result["confidence"],

    similarity=verification_result["similarity_score"],

    trust_score=trust_score,

    status=status,

    proof_hash=proof_hash

)

    # ---------------------------------------
# Module 5 - Explainable AI
# ---------------------------------------

    gradcam_path = "temp/gradcam_result.jpg"

    generate_gradcam(
    verification_image,
    gradcam_path
)   

    # ---------------------------------------
    # Final Results
    # ---------------------------------------

    print("\n========== FINAL RESULT ==========\n")

    print(f"Deepfake Prediction : {deepfake_result['prediction']}")
    print(f"Deepfake Confidence : {deepfake_result['confidence']:.2f}%")

    print()

    print(f"Face Verified       : {verification_result['verified']}")
    print(f"Similarity Score    : {verification_result['similarity_score']:.2f}%")

    print()

    print(f"Trust Score         : {trust_score:.2f}")
    print(f"Verification Status : {status}")

    print()

    print("========== ZERO KNOWLEDGE PROOF ==========\n")

    print(f"Proof Generated : {proof_generated}")
    print(f"Proof Verified  : {zkp_result['proof_verified']}")
    print(f"Policy Passed   : {zkp_result['policy_passed']}")
    
    return {
        "deepfake": deepfake_result,
        "verification": verification_result,
        "trust_score": trust_score,
        "status": status,
        "zkp": {
            "generated": proof_generated,
            "verified": zkp_result["proof_verified"],
            "policy_passed": zkp_result["policy_passed"]
            },
        "blockchain": {
           "proof_hash": proof_hash
        },
        "gradcam": {
           "image": gradcam_path
        }
    }