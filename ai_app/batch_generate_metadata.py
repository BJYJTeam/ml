import os
from ai_app.image_captioning import generate_image_metadata

def batch_generate_metadata(directory):
    metadata_list = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            filepath = os.path.join(directory, filename)
            # Generate metadata for the image file
            metadata = generate_image_metadata(filepath)
            metadata_list.append(metadata)
    return metadata_list


if __name__ == "__main__":
    image_dir = "/Users/alice/Documents/School/React/bjyj/bjyj-ml/media"
    metadata = batch_generate_metadata(image_dir)

    output_file = "medical_metadata.json"
    with open(output_file, "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Metadata for {len(metadata)} images saved to '{output_file}'")
