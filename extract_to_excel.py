import os
import re
import pandas as pd

folder_path = r"D:\Requlatory-Chatbot\eval\outputs"

data = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract sections
        answer_match = re.search(r"ANSWER:(.*?)CONFIDENCE:", content, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""

        confidence_match = re.search(r"CONFIDENCE:\s*(.*)", content)
        confidence = confidence_match.group(1).strip() if confidence_match else ""

        citations_match = re.search(r"Citations:(.*)", content, re.DOTALL)
        citations = citations_match.group(1).strip() if citations_match else ""

        data.append({
            "Answer": answer,
            "CONFIDENCE": confidence,
            "Citations": citations
        })

# Create dataframe
df = pd.DataFrame(data)

# Save to Excel
output_file = r"D:\Requlatory-Chatbot\eval\outputs\combined_results.xlsx"
df.to_excel(output_file, index=True)

print("Excel file created at:", output_file)
