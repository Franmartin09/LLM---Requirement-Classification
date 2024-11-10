import json
from classifier.model import Classification
from groq_client import client
from classifier.utils import load_json
from classifier.rewrite import rewrite_requirement

# Load department context
context = load_json("departments.json")

def get_classification(project: str):
    json_schema = json.dumps(Classification.model_json_schema(), indent=8)
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
        # Attempt to validate and return structured response using model_validate
        return Classification.model_validate_json(response_content)
    except Exception as e:
        print(f"Error during validation: {e}")
        print(f"Response content that caused the error: {response_content}")
        return None  # or handle it differently depending on the use case
    
def classify_requirements(requirements):
    """Classifies and rewrites a list of customer requirements."""
    classified_requirements = []

    for req in requirements:
        project_prompt = f"""
        Please classify the following customer requirement into one or more relevant subdepartments. 
        Be specific in your classification, do not include department name, only the subdepartment. If the requirement is relevant to only one subdepartment, only include that one. 
        If it's relevant to multiple subdepartments, list all that apply. 

        Requirement: "{req['description']}"
        """
    
        response = get_classification(project_prompt).model_dump()['classification']
        # Check if we got a valid response
        if response:
            if(len(response)>1):
                result = rewrite_requirement(req, response)
                classified_requirements.append({
                    "original_requirement": req['description'],
                    "classification_and_rewrites": response,
                    "rewrites": result
                })
            else:
                classified_requirements.append({
                    "original_requirement": req['description'],
                    "classification_and_rewrites": response
                })
        else:
            classified_requirements.append({
                "original_requirement": req['description'],
                "classification_and_rewrites": "Error in classification"
            })

    return classified_requirements
