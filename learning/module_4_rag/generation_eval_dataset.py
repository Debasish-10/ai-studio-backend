
GENERATION_EVAL_CASES = [
    {
        "id": "annual_leave",
        "question": (
            "How many annual leave days do "
            "NovaTech employees receive?"
        ),
        "reference_answer": "24 working days",
        "required_facts": [
            "24"
        ],
        "should_abstain": False
    },
    {
        "id": "carry_forward",
        "question": (
            "How many unused annual leave days "
            "can employees carry forward?"
        ),
        "reference_answer": "Up to 10 days",
        "required_facts": [
            "10"
        ],
        "should_abstain": False
    },
    {
        "id": "sick_leave",
        "question": (
            "How many sick leave days "
            "are available annually?"
        ),
        "reference_answer": "12 days",
        "required_facts": [
            "12"
        ],
        "should_abstain": False
    },
    {
        "id": "work_from_home",
        "question": (
            "How many days per week can "
            "eligible employees work remotely?"
        ),
        "reference_answer": (
            "Up to two days per week"
        ),
        "required_facts": [
            "two days"
        ],
        "should_abstain": False
    },
    {
        "id": "pet_insurance",
        "question": (
            "Does NovaTech provide pet insurance?"
        ),
        "reference_answer": (
            "The handbook does not specify "
            "pet insurance coverage."
        ),
        "required_facts": [],
        "should_abstain": True
    }
]
