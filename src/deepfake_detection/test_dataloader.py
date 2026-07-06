from src.deepfake_detection.dataloader import train_loader

def main():
    images, labels = next(iter(train_loader))

    print()
    print("Batch Shape:", images.shape)
    print("Labels Shape:", labels.shape)
    print(labels)

if __name__ == "__main__":
    main()