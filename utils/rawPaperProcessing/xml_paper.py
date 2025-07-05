import os
from dotenv import load_dotenv
import requests

load_dotenv()
Elsiver_API_KEY = os.getenv("ELSIVER_API_KEY")

def download_paper_xml(pii: str, target_path: str):
    """
    Download the raw XML version of a paper from Elsevier using its PII and save to the target path.

    Args:
        pii (str): The PII of the target paper.
        target_path (str): The file path where the XML should be saved.
    """
    url = f'https://api.elsevier.com/content/article/pii/{pii}'
    headers = {
        'X-ELS-APIKey': Elsiver_API_KEY,
        'Accept': 'application/xml'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        if response.content.strip():
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, 'wb') as file:
                file.write(response.content)
        else:
            print(f'Empty XML content received for PII: {pii}')
    else:
        print(f'Failed to retrieve article XML for PII: {pii}. Status code: {response.status_code}')