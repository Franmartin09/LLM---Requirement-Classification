import json
from classifier.model import Rewrite
from groq_client import client
from classifier.utils import load_json

# Load department context
context = load_json("departments.json")

def get_rewrite(project: str):
    json_schema = json.dumps(Rewrite.model_json_schema(), indent=8)
    system_message = f"""
    You are an AI assistant with knowledge of IT department structure. Here is the context of each department and subdepartment:
    {json.dumps(context)}
    You have to take this following JSON and only complete the gap:
    {json_schema}
    """
    
    user_message = f"{project}"

    # Get the response from Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        model="llama-3.1-70b-versatile",
        temperature=0,
        stream=False,
        response_format={"type": "json_object"},
    )

    # Inspect the raw response content
    response_content = chat_completion.choices[0].message.content

    try:
        return Rewrite.model_validate_json(response_content)
    except Exception as e:
        print(f"Error during validation: {e}")
        print(f"Response content that caused the error: {response_content}")
        return None  # or handle it differently depending on the use case

def rewrite_requirement(req: str, subdepartment: list):
    project_prompt = f"""
    Please rewrite the following customer requirement, one for each subdepartment, to be more specific.

    Requirement: "{req}"
    Subdepartments: {subdepartment}
    """

    response = get_rewrite(project_prompt)

    # Check if we got a valid response
    if response:
        return response.model_dump()['example']
    else:
        return f"Error rewriting for {subdepartment}"