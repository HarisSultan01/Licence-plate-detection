from inference_sdk import InferenceHTTPClient
import os
import json

# Initialize client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3YdMno1QtStiwIU4c39F"
)

# Get absolute image path
image_path = os.path.abspath("reading_license_plate-11/test/images/0003_jpg.rf.250a8a8bec3043e071398139c94a89d9.jpg")

# Run inference
result = CLIENT.infer(image_path, model_id="reading_license_plate/9")

# Print result
print(result)

# Save result to JSON file
output_json_path = "inference_result.json"
with open(output_json_path, "w") as json_file:
    json.dump(result, json_file, indent=4)

print(f"Result saved to {output_json_path}")
