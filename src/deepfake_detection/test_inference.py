from src.deepfake_detection.inference import predict

image_path = input("Enter Image Path: ")

result = predict(image_path)

print("\nReturned Dictionary")

print(result)