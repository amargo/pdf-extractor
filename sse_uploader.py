import requests
import json

class SSEUploader:
    """
    Handles uploading data to an SSE endpoint.
    """
    def __init__(self, url):
        """
        Initialize the SSE uploader with the target URL.
        
        Args:
            url: The URL of the SSE endpoint.
        """
        self.url = url

    def upload(self, data):
        """
        Uploads data to the SSE endpoint.

        Args:
            data: The data to upload (should be a dictionary).

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        if not self.url:
            print("SSE upload URL not provided. Skipping upload.")
            return False

        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(self.url, data=json.dumps(data), headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            print(f"Successfully uploaded data to {self.url}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error uploading data to {self.url}: {e}")
            return False