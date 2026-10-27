
EVAL_CASES = [
    {
        "id": "annual_leave",
        "query": "How many annual leave days are available?",
        "evidence_patterns": [
            r"\b24\b",
            r"annual leave"
        ]
    },
    {
        "id": "carry_forward",
        "query": "How many leave days can be carried forward?",
        "evidence_patterns": [
            r"\b10\b",
            r"carry forward"
        ]
    },
    {
        "id": "sick_leave",
        "query": "How many sick leave days are available?",
        "evidence_patterns": [
            r"\b12\b",
            r"sick leave"
        ]
    },
    {
        "id": "leave_deadline",
        "query": "When must carried-forward leave be used?",
        "evidence_patterns": [
            r"september 30",
            r"leave"
        ]
    },
    {
        "id": "remote_work",
        "query": "How often can employees work from home?",
        "evidence_patterns": [
            r"two days per week",
            r"remote"
        ]
    }
]
