import pinecone
from groq import LLaMA3API  # Suponiendo una clase ficticia para simplificar la llamada a la API de Groq para LLaMA3
from typing import List, Dict

# Inicialización de Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY", environment="us-west1-gcp")

# Inicialización del índice de Pinecone
index = pinecone.Index("requirements-classification")

# Configuración de la API de Groq para LLaMA 3
llama_api = LLaMA3API(api_key="YOUR_GROQ_API_KEY")

def rewrite_customer_requirement(customer_requirement: str) -> str:
    """Usa LLaMA3 a través de la API de Groq para reescribir el customer requirement como un system requirement."""
    prompt = f"Rewrite the following customer requirement in a way that is understandable for all departments. Customer requirement: '{customer_requirement}'"
    response = llama_api.generate(prompt)
    return response['text']  # Suponiendo que 'text' contiene el system requirement reescrito

def classify_system_requirement(system_requirement: str) -> List[str]:
    """Clasifica preliminarmente el system requirement en departamentos usando Pinecone para similitud."""
    # Crear un embedding del system requirement
    embedding = llama_api.get_embedding(system_requirement)
    
    # Buscar en Pinecone para obtener los departamentos más probables
    results = index.query(vector=embedding, top_k=3, include_metadata=True)
    
    # Obtener departamentos sugeridos en base a los resultados
    departments = []
    for match in results['matches']:
        if match['score'] > 0.8:  # Umbral de confianza
            departments.append(match['metadata']['department'])
    
    return list(set(departments))  # Eliminando duplicados

def decompose_for_departments(system_requirement: str, departments: List[str]) -> Dict[str, str]:
    """Genera versiones específicas del system requirement para cada departamento relevante usando LLaMA3."""
    decomposed_requirements = {}
    for department in departments:
        prompt = (f"Rewrite the following system requirement to include only the relevant parts for the {department} department: "
                  f"'{system_requirement}'")
        response = llama_api.generate(prompt)
        decomposed_requirements[department] = response['text']  # Suponiendo que 'text' contiene la versión específica
    return decomposed_requirements

def save_to_pinecone(system_requirement: str, departments: List[str]):
    """Guarda el embedding del requirement en Pinecone para futuras referencias."""
    embedding = llama_api.get_embedding(system_requirement)
    metadata = {"text": system_requirement, "departments": departments}
    index.upsert([(system_requirement, embedding, metadata)])

def classify_and_process(customer_requirement: str):
    """Pipeline completa para clasificar y procesar el requirement."""
    # Paso 1: Reescribir el customer requirement
    system_requirement = rewrite_customer_requirement(customer_requirement)
    print(f"System Requirement: {system_requirement}")
    
    # Paso 2: Clasificar en departamentos
    departments = classify_system_requirement(system_requirement)
    print(f"Departments: {departments}")
    
    # Paso 3: Descomponer para cada departamento relevante
    decomposed_requirements = decompose_for_departments(system_requirement, departments)
    for department, dept_requirement in decomposed_requirements.items():
        print(f"Requirement for {department} Department: {dept_requirement}")
    
    # Paso 4: Guardar el requirement en Pinecone
    save_to_pinecone(system_requirement, departments)
    print("Requirement processed and stored in Pinecone.")

# Ejemplo de uso
customer_requirement_example = "The system should withstand high temperatures without impacting performance."
classify_and_process(customer_requirement_example)
