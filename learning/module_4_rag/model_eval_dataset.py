
QUERIES = {
    "q1": "How many annual leave days are employees entitled to?",
    "q2": "How many unused vacation days can be rolled over?",
    "q3": "What is the sick leave entitlement?",
    "q4": "How often are employees allowed to work remotely?",
    "q5": "What happens to unused leave during final settlement?",
}

CORPUS = {
    "d1": (
        "Employees receive 24 working days "
        "of annual leave every year."
    ),
    "d2": (
        "Up to 10 unused annual leave days "
        "may be carried forward."
    ),
    "d3": (
        "Eligible employees receive 12 days "
        "of paid sick leave annually."
    ),
    "d4": (
        "Employees may work remotely up to "
        "two days per week with approval."
    ),
    "d5": (
        "Eligible unused annual leave may be "
        "encashed during final settlement."
    ),
    "d6": (
        "Annual leave requests require "
        "manager approval before travel."
    ),
    "d7": (
        "Employees receive health insurance "
        "under the applicable benefit plan."
    ),
    "d8": (
        "All employees must complete annual "
        "information security training."
    ),
    "d9": (
        "New employees must complete "
        "onboarding documentation."
    ),
    "d10": (
        "Unapproved absences may be considered "
        "under the attendance policy."
    ),
}

# Human-verifiable evidence labels for this toy dataset.
RELEVANT_DOCS = {
    "q1": {"d1"},
    "q2": {"d2"},
    "q3": {"d3"},
    "q4": {"d4"},
    "q5": {"d5"},
}
