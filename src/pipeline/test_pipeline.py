from src.pipeline.pipeline import run_pipeline

reference_image = input("Reference Image Path: ")

verification_image = input("Verification Image Path: ")

result = run_pipeline(
    reference_image,
    verification_image
)

print("\nReturned Dictionary\n")

print(result)