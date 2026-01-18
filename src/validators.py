def validate_source_usage(answer, docs):
    valid_files = {d["metadata"]["source_file"] for d in docs}
    for line in answer.splitlines():
        if "File:" in line:
            cited = line.split("File:")[-1].strip()
            if cited not in valid_files:
                return False
    return True

def validate_structure(answer):
    required = ["Summary", "Step-by-step", "Assumptions", "Sources", "Confidence"]
    text = answer.lower()
    return all(r.lower() in text for r in required)
