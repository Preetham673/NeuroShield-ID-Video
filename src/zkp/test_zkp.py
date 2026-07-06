from src.zkp.generate_proof import generate_proof
from src.zkp.verify_proof import verify_proof

trust_score = float(
    input("Trust Score: ")
)

print()

if generate_proof(trust_score):

    verified = verify_proof()

    print()

    print("Verified :", verified)