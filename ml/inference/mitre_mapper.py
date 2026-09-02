"""
MITRE ATT&CK Mapper
====================
Maps CIC-IDS2017 attack class labels to MITRE ATT&CK tactics,
techniques, and severity scores.

ATT&CK mapping based on:
  Ehsan et al. (2020), MITRE ATT&CK for IDS/IPS datasets.
  CIC-IDS2017 label annotations by Sharafaldin et al. (2018).
"""

# Severity: 1 (low) → 5 (critical)
# Stage order mirrors kill-chain progression

MITRE_MAPPING = {
    "BENIGN": {
        "tactic":    None,
        "technique": None,
        "tactic_id": None,
        "stage":     0,          # 0 = no attack
        "severity":  0,
        "kill_chain_phase": "None",
    },
    "Bot": {
        "tactic":    "Command and Control",
        "technique": "Application Layer Protocol",
        "tactic_id": "TA0011",
        "technique_id": "T1071",
        "stage":     6,
        "severity":  5,
        "kill_chain_phase": "command-and-control",
    },
    "DDoS": {
        "tactic":    "Impact",
        "technique": "Network Denial of Service",
        "tactic_id": "TA0040",
        "technique_id": "T1498",
        "stage":     7,
        "severity":  5,
        "kill_chain_phase": "actions-on-objectives",
    },
    "DoS_GoldenEye": {
        "tactic":    "Impact",
        "technique": "Endpoint Denial of Service",
        "tactic_id": "TA0040",
        "technique_id": "T1499",
        "stage":     7,
        "severity":  4,
        "kill_chain_phase": "actions-on-objectives",
    },
    "DoS_Hulk": {
        "tactic":    "Impact",
        "technique": "Endpoint Denial of Service",
        "tactic_id": "TA0040",
        "technique_id": "T1499",
        "stage":     7,
        "severity":  4,
        "kill_chain_phase": "actions-on-objectives",
    },
    "DoS_Slowhttptest": {
        "tactic":    "Impact",
        "technique": "Endpoint Denial of Service (Slowloris variant)",
        "tactic_id": "TA0040",
        "technique_id": "T1499.001",
        "stage":     7,
        "severity":  3,
        "kill_chain_phase": "actions-on-objectives",
    },
    "DoS_slowloris": {
        "tactic":    "Impact",
        "technique": "Endpoint Denial of Service (Slowloris)",
        "tactic_id": "TA0040",
        "technique_id": "T1499.001",
        "stage":     7,
        "severity":  3,
        "kill_chain_phase": "actions-on-objectives",
    },
    "FTP-Patator": {
        "tactic":    "Credential Access",
        "technique": "Brute Force: Password Spraying",
        "tactic_id": "TA0006",
        "technique_id": "T1110.003",
        "stage":     3,
        "severity":  3,
        "kill_chain_phase": "exploitation",
    },
    "Heartbleed": {
        "tactic":    "Initial Access",
        "technique": "Exploit Public-Facing Application",
        "tactic_id": "TA0001",
        "technique_id": "T1190",
        "stage":     2,
        "severity":  5,
        "kill_chain_phase": "delivery",
    },
    "Infiltration": {
        "tactic":    "Lateral Movement",
        "technique": "Exploitation of Remote Services",
        "tactic_id": "TA0008",
        "technique_id": "T1210",
        "stage":     5,
        "severity":  5,
        "kill_chain_phase": "lateral-movement",
    },
    "PortScan": {
        "tactic":    "Discovery",
        "technique": "Network Service Discovery",
        "tactic_id": "TA0007",
        "technique_id": "T1046",
        "stage":     1,          # earliest stage — reconnaissance
        "severity":  2,
        "kill_chain_phase": "reconnaissance",
    },
    "SSH-Patator": {
        "tactic":    "Credential Access",
        "technique": "Brute Force: Password Spraying",
        "tactic_id": "TA0006",
        "technique_id": "T1110.003",
        "stage":     3,
        "severity":  3,
        "kill_chain_phase": "exploitation",
    },
    "WebAttack_BruteForce": {
        "tactic":    "Credential Access",
        "technique": "Brute Force",
        "tactic_id": "TA0006",
        "technique_id": "T1110",
        "stage":     3,
        "severity":  3,
        "kill_chain_phase": "exploitation",
    },
    "WebAttack_SQL_Injection": {
        "tactic":    "Initial Access",
        "technique": "Exploit Public-Facing Application",
        "tactic_id": "TA0001",
        "technique_id": "T1190",
        "stage":     2,
        "severity":  5,
        "kill_chain_phase": "delivery",
    },
    "WebAttack_XSS": {
        "tactic":    "Execution",
        "technique": "Scripting: Cross-Site Scripting",
        "tactic_id": "TA0002",
        "technique_id": "T1059.007",
        "stage":     4,
        "severity":  3,
        "kill_chain_phase": "installation",
    },
}

# Severity weights for risk scoring (0–5 → normalized)
SEVERITY_WEIGHTS = {label: info["severity"] for label, info in MITRE_MAPPING.items()}

# Kill-chain stage names (ordered)
KILL_CHAIN_STAGES = [
    "None", "Reconnaissance", "Delivery", "Exploitation",
    "Installation", "Lateral Movement", "C2", "Actions on Objectives"
]

TACTIC_COLORS = {
    None:                    "#95a5a6",
    "Discovery":             "#3498db",
    "Initial Access":        "#e67e22",
    "Credential Access":     "#e74c3c",
    "Execution":             "#9b59b6",
    "Lateral Movement":      "#1abc9c",
    "Command and Control":   "#c0392b",
    "Impact":                "#922b21",
}


def get_mitre_info(class_name: str) -> dict:
    """Return MITRE ATT&CK info for a given class label."""
    return MITRE_MAPPING.get(class_name, MITRE_MAPPING["BENIGN"])


def get_next_likely_stage(current_stage: int, class_name: str) -> str:
    """
    Given a current MITRE kill-chain stage, return the most likely next stage name.
    Progression follows: Recon → Delivery → Exploitation → Installation → LM → C2 → Impact
    """
    stage_progression = {
        0: "Reconnaissance (PortScan / Network Discovery)",
        1: "Delivery (Exploit / Heartbleed / Web Attack)",
        2: "Exploitation (Credential Brute Force)",
        3: "Installation (WebAttack_XSS / Code Execution)",
        4: "Lateral Movement (Infiltration)",
        5: "Command & Control (Bot)",
        6: "Impact (DDoS / DoS)",
        7: "Post-Impact (Exfiltration / Persistence)",
    }
    next_stage = min(current_stage + 1, 7)
    return stage_progression.get(next_stage, "Unknown")
