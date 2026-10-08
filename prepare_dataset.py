import shutil

from config import (
    VERIFIED_DIR,
    DATASET_IMAGE_DIR,
    DATASET_LABEL_DIR,
)


def main():

    print("=" * 70)
    print("PREPARING FINAL DATASET")
    print("=" * 70)

    verified_images = (
        VERIFIED_DIR / "images"
    )

    verified_labels = (
        VERIFIED_DIR / "labels"
    )

    if not verified_images.exists():

        print(
            "Verified image directory does not exist."
        )

        print(
            f"Expected:\n{verified_images}"
        )

        return

    if not verified_labels.exists():

        print(
            "Verified label directory does not exist."
        )

        print(
            f"Expected:\n{verified_labels}"
        )

        return

    image_files = [
        path
        for path in verified_images.iterdir()
        if path.is_file()
    ]

    copied_images = 0
    copied_labels = 0
    missing_labels = 0

    for image_path in image_files:

        label_path = (
            verified_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: Missing label: "
                f"{image_path.name}"
            )

            missing_labels += 1

            continue

        destination_image = (
            DATASET_IMAGE_DIR
            / image_path.name
        )

        destination_label = (
            DATASET_LABEL_DIR
            / label_path.name
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        shutil.copy2(
            label_path,
            destination_label,
        )

        copied_images += 1
        copied_labels += 1

    print("\n" + "=" * 70)

    print(
        f"Images copied: {copied_images}"
    )

    print(
        f"Labels copied: {copied_labels}"
    )

    print(
        f"Missing labels: {missing_labels}"
    )

    print(
        f"\nFinal images:\n{DATASET_IMAGE_DIR}"
    )

    print(
        f"\nFinal labels:\n{DATASET_LABEL_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
