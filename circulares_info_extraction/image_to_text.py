from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from skimage import color, filters, morphology, util
from circulares_info_extraction.utils_etl import image_to_bytes
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random
from circulares_info_extraction.api_clients import textract_client
from circulares_info_extraction.config import LoadConfig

config = LoadConfig()
config.set_section('image_processing')
TEXTRACT_CONFIDENCE_FIRST_PAGE = config.parameter("textract_confidence_first_page")
TEXTRACT_CONFIDENCE = config.parameter("textract_confidence")
NOISE_THRESHOLD = config.parameter("noise_threshold")
FILTER_DISK_RATIO = config.parameter("filter_disk_ratio")
MORPHLOGYCAL_REMOVAL_RATIO = config.parameter("morphological_removal_ratio")

MEDIAN_BLUR_KERNEL_SIZE = config.parameter("median_blur_kernel_size")
KERNEL_HEIGHT = config.parameter("kernel_height")
KERNEL_WIDTH = config.parameter("kernel_width")
GAUSSIAN_KERNEL_HEIGHT = config.parameter("gaussian_kernel_height")
GAUSSIAN_KERNEL_WIDTH = config.parameter("gaussian_kernel_width")


def postprocess_extracted_text(text_list: list) -> list:
    new_text_list = [line.replace(":\n", ": ") for line in text_list]
    return new_text_list


def simple_processing_image(image):
    """
    Processes an image by converting it to grayscale, enhancing its contrast, and applying edge enhancement.
    Parameters:image (PIL.Image.Image): The image to process, expected to be a PIL image object.
    Returns: PIL.Image.Image: The processed image with enhanced contrast and edges.
    """
    # Assuming 'image' is your PIL.TiffImagePlugin.TiffImageFile object
    # Ensure the image is in grayscale mode
    if image.mode != 'L':
        image = image.convert('L')

    # Apply image processing techniques
    # Increase contrast
    contrast = ImageEnhance.Contrast(image)
    high_contrast = contrast.enhance(2)
    # Apply edge enhancement
    enhanced_image = high_contrast.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # Print extracted text
    return enhanced_image


def detect_image_noise(image):
    """
    Detects how much noise there is in an image
    @param image: image in PIL format
    @param threshold:
    @return:
    """
    image = np.array(image)
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = color.rgb2gray(image)
    else:
        gray = util.img_as_float(image)

    # Apply median filter to reduce noise and preserve edges
    blurred = filters.median(gray, footprint=morphology.disk(FILTER_DISK_RATIO))

    # Apply adaptive thresholding to segment the foreground and background
    binary = blurred > filters.threshold_otsu(blurred)

    # Apply morphological opening to remove small objects
    opened = morphology.opening(binary, morphology.disk(MORPHLOGYCAL_REMOVAL_RATIO))

    # Get the background pixels
    background_pixels = gray[~opened]

    # Calculate the standard deviation of the background pixels
    background_std = util.img_as_float(background_pixels).std()

    # If the standard deviation of the background pixels is above the threshold,
    # consider it as background noise
    return background_std * 100


def advanced_processing_image(image, image_noise):
    # Convert the page object to a NumPy array
    img_array = np.array(image)

    # Ensure the image is of type uint8
    img_array = img_array.astype(np.uint8)
    if image_noise >= NOISE_THRESHOLD:
        print(f"Image noise : {image_noise}, higher than {NOISE_THRESHOLD}, doing advanced cleaning")
        # Apply median blur to reduce noise
        median_filtered = cv2.medianBlur(img_array, MEDIAN_BLUR_KERNEL_SIZE)

        # Adaptive thresholding to binarize the image
        _, binary_image = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Morphological opening to remove small white patches and improve continuity of letters
        kernel = np.ones((KERNEL_HEIGHT, KERNEL_WIDTH), np.uint8)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        opened_image = cv2.bitwise_not(opened_image)

        # Apply Gaussian blur
        gaussian_filtered = cv2.GaussianBlur(opened_image, (GAUSSIAN_KERNEL_HEIGHT, GAUSSIAN_KERNEL_WIDTH), 0)
        enhanced_image = Image.fromarray(gaussian_filtered.astype('uint8'))
        return enhanced_image
    else:
        print(f"Image noise : {image_noise}, lower than {NOISE_THRESHOLD} no advanced cleaning")
        enhanced_image = simple_processing_image(image)
        return enhanced_image


def preprocess_image(image, advanced=True):
    """
    First there is a noise detection algorithm and next we clean the image
    @param image: image in format PIL
    @param advanced: If true make the advanced processing
    @return enhanced_image: image in format PIL after processing
    """
    if advanced:
        image_noise = detect_image_noise(image)
        enhanced_image = advanced_processing_image(image, image_noise)
    else:
        enhanced_image = simple_processing_image(image)

    return enhanced_image


def process_image(img, confidence_threshold=TEXTRACT_CONFIDENCE, index=None):
    """
    Process a single image: preprocess and extract text.
    """
    try:
        # Simple processing first
        print(f"Preprocessing page {index} with simple processing")
        enhanced_image = preprocess_image(img, advanced=False)

        print(f"Extracting text with AWS Textract from page {index}")
        text, page_confidence = extract_text_with_textract(enhanced_image)

        # Get average confidence
        print(f"Average confidence for page {index}: {page_confidence}")

        # If confidence is below the threshold, use advanced processing
        if page_confidence < confidence_threshold:
            print(f"Reprocessing page {index} with advanced processing due to low confidence")
            enhanced_image = preprocess_image(img, advanced=True)
            text, page_confidence = extract_text_with_textract(enhanced_image)
            print(f"Average confidence for page {index} after processing: {page_confidence}")
        return index, text, enhanced_image

    except Exception as e:
        print(f"Error processing image at index {index}: {e}")
        # Return index and placeholders for text and enhanced_image in case of error
        return index, "", None


@retry(stop=stop_after_attempt(10), wait=wait_fixed(2) + wait_random(0, 2))
def extract_text_with_textract(enhanced_image):
    """
    Extract text using AWS Textract with built-in retry logic for handling exceptions.
    """
    image_bytes = image_to_bytes(enhanced_image)
    response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
    extracted_text = "\n".join(block['Text'] for block in response['Blocks'] if block['BlockType'] == 'LINE')

    confidences = [
        block["Confidence"]
        for block in response["Blocks"]
        if block["BlockType"] == "LINE"
    ]
    avg_confidence = np.mean(confidences)

    return extracted_text, avg_confidence


def extract_text_from_images(imgs, confidence_threshold=TEXTRACT_CONFIDENCE):
    """
    Extracts text from a list of images using Amazon Textract or Tesseract in parallel.
    Maintains the order of the input images in the output lists.
    Handles single and multiple images correctly.
    """
    # Initialize placeholders for results
    results = [None] * len(imgs)

    with ThreadPoolExecutor() as executor:
        # Schedule the process_image function to be executed for each image
        futures = [executor.submit(process_image, img, confidence_threshold, index) for index, img in
                   enumerate(imgs)]

        # Wait for all futures to complete and store results in their respective index
        for future in futures:
            index, text, enhanced_image = future.result()
            results[index] = (text, enhanced_image)

    # Unpack results into separate lists while preserving order
    text_list, enhanced_images_list = zip(*results) if results else ([], [])
    text_list = postprocess_extracted_text(text_list)

    return list(text_list), list(enhanced_images_list)
