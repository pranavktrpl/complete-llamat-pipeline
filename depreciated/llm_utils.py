from .callLLM import text_completion, structured_completion, StructuredRequest, CompletionRequest
import os
import json

# with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_in_context.txt")) as f:
#     complete_the_table_w_in_context_prompt = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_in_context_for_schema.txt")) as f:
#     complete_the_table_w_in_context_schema_prompt = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompts/complete_the_table_w_out_in_context.txt")) as f:
#     complete_the_table_w_out_in_context_prompt = f.read()


# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/Matskraft-tables/S0167577X06001327.txt")) as f:
#     incomplete_table = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/research-paper-tables/S0167577X06001327.txt")) as f:
#     research_paper_tables = f.read()

# with open(os.path.join(os.path.dirname(__file__), "../prompting_data/research-paper-text/S0167577X06001327.txt")) as f:
#     research_paper_text = f.read()


##################################################################################################################
with open(os.path.join(os.path.dirname(__file__), "../prompts/fluid-context-control.txt")) as f:
    fluid_incontext_control = f.read()


def fluid_incontext_prompts(model: str, temperature: float, in_context_examples: str, research_paper_context:str, incomplete_table: str):
    prompt = fluid_incontext_control.replace("{{incontextExamples}}", in_context_examples)
    prompt = prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    
    response_text, metadata = text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000
    ))

    # Clean up the response text to remove markdown code block syntax
    if isinstance(response_text, str):
        cleaned_response = response_text.replace('```json\n', '')
        cleaned_response = cleaned_response.replace('```', '')
        cleaned_response = cleaned_response.strip()
        
        try:
            # Parse and re-serialize to ensure clean JSON
            response_json = json.loads(cleaned_response)
            
            # Extract relevant metadata
            cleaned_metadata = {
                "promptTokenCount": metadata.get("promptTokenCount", 0),
                "candidatesTokenCount": metadata.get("candidatesTokenCount", 0),
                "totalTokenCount": metadata.get("totalTokenCount", 0)
            }
            
            return response_json, cleaned_metadata
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
    
    return None, metadata


def geminifluid_incontext_prompts(model: str, temperature: float, in_context_examples: str, research_paper_context:str, incomplete_table: str):
    prompt = fluid_incontext_control.replace("{{incontextExamples}}", in_context_examples)
    prompt = prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    
    response_text, metadata = text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=100000
    ))

    print(response_text)
    # Clean up the response text to remove markdown code block syntax
    if isinstance(response_text, str):
        response_text = response_text.replace('```json\n', '')
        response_text = response_text.replace('```', '')
        response_text = response_text.strip()
        
        try:
            # Parse and re-serialize to ensure clean JSON
            response_json = json.loads(response_text)
            
            # Extract relevant metadata
            cleaned_metadata = {
                "promptTokenCount": metadata.get("promptTokenCount", 0),
                "candidatesTokenCount": metadata.get("candidatesTokenCount", 0),
                "totalTokenCount": metadata.get("totalTokenCount", 0)
            }
            
            return response_json, cleaned_metadata
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
    
    return response_text, metadata


# context = f"{research_paper_text}\n\n{research_paper_tables}"


# def complete_the_table_in_context(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
#     prompt = complete_the_table_w_in_context_schema_prompt.replace("{{Research Paper}}", research_paper_context)
#     prompt = prompt.replace("{{Table}}", incomplete_table)
    
#     return structured_completion(StructuredRequest(
#         model=model,
#         text=prompt,
#         temperature=temperature,
#         schema={
#             "Compositions": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "placeholder": {"type": "string", "description": "Placeholder in the table (e.g., <blank_1>)"},
#                         "composition": {"type": "string", "description": "Extracted composition corresponding to the placeholder"}
#                     },
#                     "required": ["placeholder", "composition"]
#                 }
#             }
#         }
#     ))

# def complete_the_table_in_context_no_schema(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
#     prompt = complete_the_table_w_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
#     prompt = prompt.replace("{{Table}}", incomplete_table)
#     return text_completion(CompletionRequest(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         max_tokens=100000
#     ))

# def complete_the_table(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
#     prompt = complete_the_table_w_out_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
#     prompt = prompt.replace("{{Table}}", incomplete_table)
    
#     return structured_completion(StructuredRequest(
#         model=model,
#         text=prompt,
#         temperature=temperature,
#         schema={
#             "Compositions": {
#                 "type": "array",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "placeholder": {"type": "string", "description": "Placeholder in the table (e.g., <blank_1>)"},
#                         "composition": {"type": "string", "description": "Extracted composition corresponding to the placeholder"}
#                     },
#                     "required": ["placeholder", "composition"]
#                 }
#             }
#         }
#     ))


# def complete_the_table_no_schema(model: str, temperature: float, research_paper_context: str, incomplete_table: str):
#     prompt = complete_the_table_w_out_in_context_prompt.replace("{{Research Paper}}", research_paper_context)
#     prompt = prompt.replace("{{Table}}", incomplete_table)
#     return text_completion(CompletionRequest(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=temperature,
#         max_tokens=100000
#     ))


def simple_prompt(model: str, temperature: float, prompt: str):
    return text_completion(CompletionRequest(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=10000
    ))


# print(simple_prompt("gpt-4o-mini", 0, "Hi, just checking my api, are you working?" ))
# print(simple_prompt("anthropic/claude-3-haiku-20240307", 0, "Hi, just checking my api, are you working?" ))
# print(simple_prompt("anthropic/claude-3-5-haiku-20241022", 0, "Hi, just checking my api, are you working?" ))
# print(complete_the_table_in_context_no_schema("custom/gemini-flash", 0.0, context, incomplete_table))
# print(complete_the_table_in_context_no_schema("custom/gemini-pro", 0.0, context, incomplete_table))
# print(complete_the_table_no_schema("groq/llama-3.3-70b-versatile", 0.0, context, incomplete_table))
# print(complete_the_table_in_context("custom/gemini-flash", 0.0, context, incomplete_table))
# print(complete_the_table_in_context("custom/gemini-pro", 0.0, context, incomplete_table))

def llamat_fluid_incontext_prompts(model: str, temperature: float, in_context_examples: str, research_paper_context:str, incomplete_table: str):
    from .llamat_prompter import llamatPrompter
    
    # Initialize the prompter
    prompter = llamatPrompter()
    
    # Load the base prompt
    with open(os.path.join(os.path.dirname(__file__), "../prompts/llamat_complete_prompt.txt")) as f:
        base_prompt = f.read()
    
    # Replace placeholders in the prompt
    if in_context_examples:
        # If in-context examples are provided, include the Example Processing section
        prompt = base_prompt.replace("{{incontextExamples}}", in_context_examples)
    else:
        # If no in-context examples, remove the entire Example Processing section
        prompt = base_prompt.replace("Example Processing:\n<examples>\n{{incontextExamples}}\n</examples>\n", "")
    
    prompt = prompt.replace("{{Research Paper}}", research_paper_context)
    prompt = prompt.replace("{{Table}}", incomplete_table)
    
    # Get response from Llamat
    response_text = prompter(prompt)
    
    # Clean up the response text to remove markdown code block syntax
    if isinstance(response_text, str):
        response_text = response_text.replace('```json\n', '')
        response_text = response_text.replace('```', '')
        response_text = response_text.strip()
        
        # Since Llamat doesn't provide token counts, we'll use placeholder metadata
        metadata = {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0
        }
        
        return response_text, metadata
    
    return None, {}