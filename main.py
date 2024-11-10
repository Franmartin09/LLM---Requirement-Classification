from classifier.classification import classify_requirements
import json
import os
from datetime import datetime
from classifier.utils import load_json
import sys

# Define customer requirements
requirements = load_json("requirements.json")

# Run classification and get results
results = classify_requirements(requirements['requirements'])

# Create the output directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Generate a timestamp for the filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file_path = os.path.join(output_dir, f"{timestamp}.txt")

# Save the results in the output file
with open(output_file_path, "w") as output_file:
    for result in results:
        output_file.write("Original Requirement: " + result["original_requirement"] + "\n")
        output_file.write("Classification: " + json.dumps(result["classification_and_rewrites"], indent=2) + "\n")
        if "rewrites" in result:
            output_file.write("Rewrites: " + json.dumps(result["rewrites"], indent=2) + "\n")
        output_file.write("\n" + "="*50 + "\n")

# Write the completion message
with open(output_file_path, "a") as output_file:
    output_file.write("\nProcess completed successfully.\n")

print("Results saved to:", output_file_path)
