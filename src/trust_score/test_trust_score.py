from src.trust_score.trust_score import calculate_trust_score

deepfake_confidence = float(
    input("Deepfake Confidence (0-100): ")
)

similarity_score = float(
    input("Similarity Score (0-100): ")
)

score, status = calculate_trust_score(
    deepfake_confidence,
    similarity_score
)

print("\n========== TRUST SCORE ==========\n")

print(f"Trust Score : {score:.2f}")

print(f"Status      : {status}")