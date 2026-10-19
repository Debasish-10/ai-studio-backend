"""Ground-truth retrieval questions for NovaTech's fictional employee handbook.

Based on the actual chunk IDs from the user's inspected ChromaDB corpus.

`relevant_chunk_ids` are evidence-bearing chunks, not merely topic-related chunks.
For repeated information (e.g., main policy + FAQ), multiple IDs are listed.
The simple Recall@K evaluator counts every listed ID separately; therefore its
score can penalize a system that retrieves just one perfectly sufficient source.
Multi-fact questions also require evidence coverage beyond simple Hit@K.

These labels are a manually curated starting point, not exhaustive gold labels.
"""

EVALUATION_DATASET = [
    # Direct factual retrieval
    {
        "id": "q01", "category": "factual",
        "query": "What is the standard working week for full-time employees in India?",
        "relevant_chunk_ids": ["handbook_001_p1_c2"],
    },
    {
        "id": "q02", "category": "factual",
        "query": "How many paid annual leave days do employees receive each year?",
        "relevant_chunk_ids": ["handbook_001_p2_c2", "handbook_001_p9_c3"],
    },
    {
        "id": "q03", "category": "factual",
        "query": "How many unused annual leave days can be carried forward, and when must they be used?",
        "relevant_chunk_ids": ["handbook_001_p2_c4", "handbook_001_p9_c3"],
    },
    {
        "id": "q04", "category": "factual",
        "query": "How many sick leave days do Indian employees receive?",
        "relevant_chunk_ids": ["handbook_001_p3_c1", "handbook_001_p3_c2", "handbook_001_p9_c4"],
    },
    {
        "id": "q05", "category": "factual",
        "query": "When is a medical certificate potentially required for sick leave?",
        "relevant_chunk_ids": ["handbook_001_p3_c2"],
    },
    {
        "id": "q06", "category": "factual",
        "query": "How many casual leave days are employees entitled to annually?",
        "relevant_chunk_ids": ["handbook_001_p3_c3"],
    },
    {
        "id": "q07", "category": "factual",
        "query": "How many paid public holidays do employees in India receive?",
        "relevant_chunk_ids": ["handbook_001_p3_c4"],
    },
    {
        "id": "q08", "category": "factual",
        "query": "What is the maximum paid maternity leave duration?",
        "relevant_chunk_ids": ["handbook_001_p4_c1"],
    },
    {
        "id": "q09", "category": "factual",
        "query": "How long is paid paternity or partner leave, and when should it be used?",
        "relevant_chunk_ids": ["handbook_001_p4_c3", "handbook_001_p4_c4"],
    },
    {
        "id": "q10", "category": "factual",
        "query": "What is the annual family-floater medical insurance coverage limit?",
        "relevant_chunk_ids": ["handbook_001_p5_c1"],
    },
    {
        "id": "q11", "category": "factual",
        "query": "How many confidential counseling sessions are available annually?",
        "relevant_chunk_ids": ["handbook_001_p5_c4"],
    },
    {
        "id": "q12", "category": "factual",
        "query": "What is the one-time home-office equipment allowance?",
        "relevant_chunk_ids": ["handbook_001_p6_c1"],
    },
    {
        "id": "q13", "category": "factual",
        "query": "What is the monthly home internet reimbursement limit?",
        "relevant_chunk_ids": ["handbook_001_p6_c2", "handbook_001_p9_c4", "handbook_001_p9_c5"],
    },
    {
        "id": "q14", "category": "factual",
        "query": "What is the daily business-travel meal reimbursement limit within India?",
        "relevant_chunk_ids": ["handbook_001_p6_c4"],
    },
    {
        "id": "q15", "category": "factual",
        "query": "When are the company's formal performance reviews held?",
        "relevant_chunk_ids": ["handbook_001_p7_c3"],
    },
    {
        "id": "q16", "category": "factual",
        "query": "What is the resignation notice period for a confirmed employee?",
        "relevant_chunk_ids": ["handbook_001_p8_c4", "handbook_001_p10_c1"],
    },

    # Paraphrases and lexical-vs-semantic challenges
    {
        "id": "q17", "category": "paraphrase",
        "query": "How often can I telecommute instead of going to the office?",
        "relevant_chunk_ids": ["handbook_001_p1_c4", "handbook_001_p9_c4"],
    },
    {
        "id": "q18", "category": "paraphrase",
        "query": "Can leftover vacation days roll over into next year?",
        "relevant_chunk_ids": ["handbook_001_p2_c4", "handbook_001_p9_c3"],
    },
    {
        "id": "q19", "category": "paraphrase",
        "query": "Does the company pay for technical certifications and courses?",
        "relevant_chunk_ids": ["handbook_001_p7_c1", "handbook_001_p7_c2", "handbook_001_p9_c5"],
    },
    {
        "id": "q20", "category": "paraphrase",
        "query": "Can I cash out unused sick leave when I leave the company?",
        "relevant_chunk_ids": ["handbook_001_p9_c2"],
    },

    # Exact policy identifier search
    {
        "id": "q21", "category": "exact_identifier",
        "query": "What does policy HR-2026-LV-042 say about annual leave entitlement?",
        "relevant_chunk_ids": ["handbook_001_p2_c2"],
    },
    {
        "id": "q22", "category": "exact_identifier",
        "query": "Explain policy HR-2026-EXP-072 and its monthly reimbursement amount.",
        "relevant_chunk_ids": ["handbook_001_p6_c2"],
    },
    {
        "id": "q23", "category": "exact_identifier",
        "query": "What security controls are required under IT-2026-SEC-091?",
        "relevant_chunk_ids": ["handbook_001_p8_c1"],
    },
    {
        "id": "q24", "category": "exact_identifier",
        "query": "What does IT-2026-AI-092 prohibit regarding company data and public AI tools?",
        "relevant_chunk_ids": ["handbook_001_p8_c2", "handbook_001_p8_c3"],
    },

    # Multi-fact evidence coverage
    {
        "id": "q25", "category": "multi_fact",
        "query": "Compare annual leave and sick leave entitlements in India.",
        "relevant_chunk_ids": ["handbook_001_p2_c2", "handbook_001_p3_c1", "handbook_001_p3_c2", "handbook_001_p9_c3", "handbook_001_p9_c4"],
    },
    {
        "id": "q26", "category": "multi_fact",
        "query": "What are the work-from-home rules and related equipment and internet benefits?",
        "relevant_chunk_ids": ["handbook_001_p1_c4", "handbook_001_p6_c1", "handbook_001_p6_c2", "handbook_001_p9_c4", "handbook_001_p9_c5"],
    },
    {
        "id": "q27", "category": "multi_fact",
        "query": "Compare the wellness reimbursement limit with the medical insurance coverage limit.",
        "relevant_chunk_ids": ["handbook_001_p5_c1", "handbook_001_p5_c2", "handbook_001_p5_c3"],
    },
    {
        "id": "q28", "category": "multi_fact",
        "query": "Summarize the notice-period requirements for probationary and confirmed employees.",
        "relevant_chunk_ids": ["handbook_001_p8_c4", "handbook_001_p10_c1"],
    },
]
