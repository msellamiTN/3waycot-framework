Generate a Chain-of-Thought for :  {query_user}

For each step:1. Describe your reasoning2. List assumptions made3. Indicate confidence level4. Identify exceptions or caveatsOutput your response in JSON format matching the following schema: [
    {
        "Steps": "Step 1",
        "reasoning": [
            "string"
        ],
        "Assumptions": [
            "string"
        ],
        "Confidence": "high or medium or low",
        "Exceptions": [
            "string"
        ],
        "Step Final Answer": "string"
    },....{
        "Steps": "Step N",
        "reasoning": [
            "string"
        ],
        "Assumptions": [
            "string"
        ],
        "Confidence": "high or medium or low",
        "Exceptions": [
            "string"
        ],
        "Step Final Answer": "string" 
        "Step Comments": "string"
    },
    {
        "Final Answer":String","Confidence": "high or medium or low"
    }       
]