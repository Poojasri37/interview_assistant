import re

def extract_skills_and_domain(text: str):
    """
    Extracts skills and predicts domain from resume text.
    Supports CS, ECE, EEE, Mechanical, Civil, and related fields.
    Returns: (skills: list[str], domain: str)
    """

    text_lower = text.lower()

    # Master skill set across domains
    skill_keywords = {
        "Computer Science": [
            "python", "java", "c++", "c#", "javascript", "html", "css", "react", "node", "angular",
            "sql", "mysql", "mongodb", "docker", "kubernetes", "aws", "azure", "gcp",
            "tensorflow", "pytorch", "machine learning", "deep learning", "nlp",
            "data analysis", "data science", "cloud", "devops", "linux", "git", "api"
        ],
        "Electronics & Communication (ECE)": [
            "vhdl", "verilog", "fpga", "embedded", "microcontroller", "arduino", "raspberry pi",
            "matlab", "digital signal processing", "dsp", "communication systems", "antenna",
            "rf", "analog", "digital electronics"
        ],
        "Electrical (EEE)": [
            "power systems", "switchgear", "transformer", "circuit breaker", "electric machines",
            "scada", "protection systems", "renewable energy", "hvac", "pcb design"
        ],
        "Mechanical": [
            "cad", "catia", "solidworks", "ansys", "autocad", "manufacturing", "thermodynamics",
            "fluid mechanics", "mechatronics", "robotics", "hvac", "cam", "fea"
        ],
        "Civil": [
            "structural analysis", "autocad", "staad pro", "revit", "surveying", "concrete technology",
            "construction management", "geotechnical", "hydraulics", "building design"
        ]
    }

    skills_found = []
    domain_scores = {domain: 0 for domain in skill_keywords}

    # Check skills and assign domain score
    for domain, keywords in skill_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                skills_found.append(keyword)
                domain_scores[domain] += 1

    # Predict domain with max score
    predicted_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[predicted_domain] == 0:
        predicted_domain = "General Engineering"

    # Deduplicate skills
    skills_found = list(set(skills_found))

    return skills_found, predicted_domain
