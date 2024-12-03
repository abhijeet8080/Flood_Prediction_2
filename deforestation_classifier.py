from inference_sdk import InferenceHTTPClient

def classify_deforestation(image_path):
    """
    Classifies the given image for deforestation.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: The top class from the API response.
    """
    # Create an inference client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Em1eW29WvXGo3RRftSuW"
    )

    # Run inference
    response = client.infer(
        image_path, 
        model_id="deforestation-nqtwb/1"
    )

    # Extract the top class
    top_class = response.get("top", "No class found")
    return top_class
