"""
Helper functions and constants for AI text processing.
"""
import os
import re

# Model name constants
MODEL_GPT4 = "gpt-4"
MODEL_GPT35_TURBO = "gpt-3.5-turbo"

# Model prefix constants
PREFIX_CLAUDE = "claude-"
PREFIX_ANTHROPIC = "anthropic/"
PREFIX_O = "o"
PREFIX_OPENAI = "openai/"

def update_model_flags(model):
    """
    Update model type flags based on the model name.
    
    Args:
        model: The model name string
        
    Returns:
        tuple: (is_claude_model, is_o_series, is_gpt_model)
    """
    is_claude_model = model.startswith(PREFIX_CLAUDE) or model.startswith(PREFIX_ANTHROPIC)
    is_o_series = model.startswith(PREFIX_O) and not model.startswith(PREFIX_OPENAI)
    is_gpt_model = 'gpt' in model.lower()
    
    return is_claude_model, is_o_series, is_gpt_model

def prepare_text_and_images(native_txt_pages, ocr_txt_pages, images):
    """
    Prepare text and image references for processing.
    
    Args:
        native_txt_pages: List of text pages from native extraction
        ocr_txt_pages: List of text pages from OCR extraction
        images: Dictionary mapping page numbers to lists of image filenames
        
    Returns:
        tuple: (total_pages, full_native_text, full_ocr_text, all_image_refs)
    """
    total_pages = min(len(native_txt_pages), len(ocr_txt_pages))
    all_native_text = []
    all_ocr_text = []
    all_image_refs = []
    
    # Collect all text and image references
    for i in range(total_pages):
        all_native_text.append(f"--- Page {i+1} ---\n{native_txt_pages[i]}\n--- End of Page {i+1} ---")
        all_ocr_text.append(f"--- Page {i+1} ---\n{ocr_txt_pages[i]}\n--- End of Page {i+1} ---")
        
        # Get image references for this page
        if i+1 in images and images[i+1]:
            page_images = []
            for img in images[i+1]:
                page_images.append(f"[IMAGE: {img}]")
            all_image_refs.append(f"--- Images on page {i+1} ---\n" + "\n".join(page_images) + "\n--- End of images ---")
    
    # Combine all text
    full_native_text = "\n\n".join(all_native_text)
    full_ocr_text = "\n\n".join(all_ocr_text)
    
    return total_pages, full_native_text, full_ocr_text, all_image_refs

def prepare_chunk(native_txt_pages, ocr_txt_pages, chunk_start, chunk_end):
    """
    Prepare text for a specific chunk of pages.
    
    Args:
        native_txt_pages: List of text pages from native extraction
        ocr_txt_pages: List of text pages from OCR extraction
        chunk_start: Starting page index
        chunk_end: Ending page index
        
    Returns:
        tuple: (full_native_chunk, full_ocr_chunk)
    """
    chunk_native = []
    chunk_ocr = []
    for i in range(chunk_start, chunk_end):
        chunk_native.append(f"--- Page {i+1} ---\n{native_txt_pages[i]}\n--- End of Page {i+1} ---")
        chunk_ocr.append(f"--- Page {i+1} ---\n{ocr_txt_pages[i]}\n--- End of Page {i+1} ---")
    
    return "\n\n".join(chunk_native), "\n\n".join(chunk_ocr)

def save_prompt_to_file(prompt, out_dir, prompt_filename):
    """
    Save the prompt to a file for debugging.
    
    Args:
        prompt: The prompt text
        out_dir: Output directory
        prompt_filename: Filename for the prompt
    """
    prompt_filepath = os.path.join(out_dir, prompt_filename)
    with open(prompt_filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Saved prompt to file: {prompt_filepath}")

def write_output_file(text, out_dir, filename="exported-ai.txt"):
    """
    Write text to output file and return the file path.
    
    Args:
        text: Text to write
        out_dir: Output directory
        filename: Output filename
        
    Returns:
        str: Path to the created file
    """
    file_path = os.path.join(out_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Created AI text file: {file_path}")
    return file_path

def print_token_usage_summary(model, is_o_series, total_tokens_used, completion_tokens_used):
    """
    Print token usage summary and estimate cost.
    
    Args:
        model: The model name
        is_o_series: Whether the model is an O-series model
        total_tokens_used: Total tokens used
        completion_tokens_used: Completion tokens used
    """
    print("\nToken usage summary:")
    print(f"Total tokens used: {total_tokens_used}")
    print(f"Completion tokens used: {completion_tokens_used}")
    print(f"Prompt tokens used: {total_tokens_used - completion_tokens_used}")
    
    # Estimate cost (very rough approximation)
    prompt_tokens_cost = 0
    completion_tokens_cost = 0
    
    if MODEL_GPT4 in model:
        prompt_tokens_cost = 0.00003 * (total_tokens_used - completion_tokens_used)
        completion_tokens_cost = 0.00006 * completion_tokens_used
    elif MODEL_GPT35_TURBO in model:
        prompt_tokens_cost = 0.0000015 * (total_tokens_used - completion_tokens_used)
        completion_tokens_cost = 0.000002 * completion_tokens_used
    elif is_o_series:
        prompt_tokens_cost = 0.000005 * (total_tokens_used - completion_tokens_used)
        completion_tokens_cost = 0.000015 * completion_tokens_used
    else:
        prompt_tokens_cost = 0.00001 * (total_tokens_used - completion_tokens_used)
        completion_tokens_cost = 0.00002 * completion_tokens_used
        
    total_cost = prompt_tokens_cost + completion_tokens_cost
    print(f"Estimated cost: ${total_cost:.6f} (input: ${prompt_tokens_cost:.6f}, output: ${completion_tokens_cost:.6f})")

def adjust_chunk_size(usage, max_input_tokens, chunk_size, chunk_end, total_pages):
    """
    Dynamically adjust chunk size based on token usage.
    
    Args:
        usage: Token usage information
        max_input_tokens: Maximum input tokens allowed
        chunk_size: Current chunk size
        chunk_end: End of current chunk
        total_pages: Total pages in document
        
    Returns:
        int: New chunk size
    """
    if chunk_end < total_pages:
        if usage.total_tokens < max_input_tokens * 0.8:
            # If we're using less than 80% of the limit, increase chunk size
            new_chunk_size = min(chunk_size + 1, 10)  # Cap at 10 pages
            if new_chunk_size != chunk_size:
                chunk_size = new_chunk_size
                print(f"Increasing chunk size to {chunk_size} pages for next chunk")
        elif usage.total_tokens > max_input_tokens * 0.9:
            # If we're close to the limit, decrease chunk size
            new_chunk_size = max(1, chunk_size - 1)
            if new_chunk_size != chunk_size:
                chunk_size = new_chunk_size
                print(f"Decreasing chunk size to {chunk_size} pages for next chunk")
    return chunk_size
