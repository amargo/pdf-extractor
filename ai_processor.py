import os
import time
import re

# Optional imports - don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Token counting function - tiktoken is preferred but we'll have a fallback
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Default OpenAI model to use
DEFAULT_OPENAI_MODEL = "o4-mini"

class AIProcessor:
    """
    Handles AI-based processing of extracted PDF text using OpenAI API.
    Creates an enhanced version combining native and OCR text.
    """
    
    def __init__(self, api_key=None, model=None):
        """
        Initialize the AI processor with API key and model.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: OpenAI model to use. If None, will use default.
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model = model or os.environ.get('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL
        
        # Determine model type
        self.is_claude_model = self.model.startswith('claude-') or self.model.startswith('anthropic/')
        self.is_o_series = self.model.startswith('o') and not self.model.startswith('openai/')
        self.is_gpt_model = 'gpt' in self.model.lower()
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get the encoding for the specific model
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fall back to cl100k_base encoding which works for most newer models
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.prompt_template = """You are an expert at merging two PDF text extractions into one clean version. {chunk_info}

1. Native Extraction: preserves original formatting.
2. OCR Extraction: may contain recognition errors.

Task: merge page by page so that the final text:
- uses the native extraction as the base, filling in gaps and fixing OCR errors from the OCR version,
- does not shorten or summarize—include every detail,
- preserves the `--- Page X ---` and `--- End of Page X ---` markers,
- keeps the `--- Images on page X ---` … `--- End of images ---` blocks and all `[IMAGE: …]` lines exactly as they appear.

--- NATIVE EXTRACTION ---
{native_text}

--- OCR EXTRACTION ---
{ocr_text}
"""
    
    def is_available(self):
        """
        Check if OpenAI processing is available (package installed and API key provided).
        
        Returns:
            bool: True if available, False otherwise
        """
        if not OPENAI_AVAILABLE:
            print("OpenAI package not installed. Skipping AI-friendly version creation.")
            print("To enable this feature, install the openai package: pip install openai")
            return False
            
        if not self.api_key:
            print("OpenAI API key not found. Skipping AI-friendly version creation.")
            print("To enable this feature, provide an API key via:")
            print("  - Command line: --openai-api-key YOUR_KEY")
            print("  - Environment variable: OPENAI_API_KEY=YOUR_KEY")
            print("  - .env file: OPENAI_API_KEY=YOUR_KEY")
            return False
            
        return True
        
    def _count_tokens(self, text):
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        if self.tokenizer is not None:
            # Use tiktoken for accurate token count
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to approximate token count (4 chars ≈ 1 token)
            # This is a rough approximation and may not be accurate for all languages
            return len(text) // 4

    def _get_api_params(self, prompt):
        """
        Constructs the parameters dictionary for an OpenAI API call.
        """
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that cleans and formats text extracted from PDFs."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add additional parameters for GPT models
        params["presence_penalty"] = 0.0
        params["stream"] = False

        # Only add temperature for models that support it (not o4-mini)
        if not self.model.startswith('o4-mini'):
            params["top_p"] = 0.9
            params["frequency_penalty"] = 0.2
            params["temperature"] = 0.7
        
        # Add the appropriate token parameter based on model type
        elif self.is_o_series:
            params["max_completion_tokens"] = 12000
        else:
            params["max_tokens"] = 16000
                        
        return params
    
    def _call_api_with_retry(self, client, prompt, out_dir, prompt_filename, fallback_text):
        """
        Calls the OpenAI API with retry logic, saves the prompt, and returns the cleaned text.
        """
        # Save the prompt to a file for debugging
        prompt_filepath = os.path.join(out_dir, prompt_filename)
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Saved prompt to file: {prompt_filepath}")

        # Calculate token count for the prompt
        prompt_token_count = self._count_tokens(prompt)
        print(f"Sending prompt with {prompt_token_count} tokens ({len(prompt)} characters) to the AI for processing...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                params = self._get_api_params(prompt)
                
                response = client.chat.completions.create(**params)
                
                # Store the response on the client object for token tracking
                if not hasattr(client, 'last_response'):
                    setattr(client, 'last_response', response)
                else:
                    client.last_response = response
                
                print(f"API Response received for {prompt_filename}")
                
                # Log token usage if available
                if hasattr(response, 'usage'):
                    print(f"Token usage: {response.usage.total_tokens} total tokens")
                    print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
                    print(f"  - Completion tokens: {response.usage.completion_tokens}")
                
                if not response.choices:
                    print("WARNING: No choices returned in the API response!")
                    continue 

                for i, choice in enumerate(response.choices):
                    print(f"Choice {i+1} details:")
                    print(f"  - Finish reason: {choice.finish_reason}")
                    if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content') and choice.message.content:
                        content_preview = choice.message.content[:100] + '...'
                        print(f"  - Content preview: {content_preview}")

                cleaned_text = response.choices[0].message.content or ""

                if response.choices[0].finish_reason == 'length':
                    print(f"Warning: AI response for {prompt_filename} was truncated due to length.")

                if cleaned_text.strip():
                    return cleaned_text

            except openai.APIError as e:
                wait_time = 2 ** attempt
                print(f"OpenAI API error on {prompt_filename}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"An unexpected error occurred on {prompt_filename}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        print(f"AI processing failed for {prompt_filename} after multiple retries. Falling back to provided text.")
        return fallback_text

    def create_ai_friendly_version(self, native_txt_pages, ocr_txt_pages, images, out_dir):
        """
        Create an AI-friendly version of the extracted text using OpenAI API.
        This combines the native and OCR text to create a more accurate and structured version.
        
        Args:
            native_txt_pages: List of text pages from native extraction
            ocr_txt_pages: List of text pages from OCR extraction
            images: Dictionary mapping page numbers to lists of image filenames
            out_dir: Output directory for the AI-friendly version
            
        Returns:
            str: Path to the created AI-friendly file, or None if failed
        """
        if not self.is_available():
            return None
        
        try:
            print(f"Creating AI-friendly version using OpenAI API with model '{self.model}'...")
            # Create OpenAI client (compatible with openai>=1.0.0)
            client = openai.OpenAI(api_key=self.api_key)
            
            # Print model type information for debugging
            model_types = []
            if self.is_o_series:
                model_types.append("OpenAI o-series")
                print(f"Using o-series model parameters (max_completion_tokens)")
            elif self.is_gpt_model:
                model_types.append("GPT")
                print(f"Using standard model parameters (max_tokens)")
            elif self.is_claude_model:
                model_types.append("Claude")
                print(f"Using standard model parameters (max_tokens)")
            else:
                print(f"Using standard model parameters (max_tokens)")
                
            if model_types:
                print(f"Detected model type: {', '.join(model_types)}")
            else:
                print("Using default model parameters")
            
            # Prepare combined text with image references
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
            
            # Accurately calculate token counts using our token counting method
            prompt_tokens = self._count_tokens(self.prompt_template)
            native_tokens = self._count_tokens(full_native_text)
            ocr_tokens = self._count_tokens(full_ocr_text)
            total_tokens = prompt_tokens + native_tokens + ocr_tokens
            
            # Define token limits based on the model
            if self.is_o_series:
                max_input_tokens = 128000  # o4 models have large context windows
            elif self.is_gpt_model and "gpt-4" in self.model:
                max_input_tokens = 8000  # GPT-4 has a larger context window
            else:
                max_input_tokens = 6000  # Conservative limit for most models
                
            print(f"Estimated tokens: {total_tokens} (prompt: {prompt_tokens}, native: {native_tokens}, OCR: {ocr_tokens})")
            print(f"Token limit for model {self.model}: {max_input_tokens}")
            
            # Determine if we need to process in chunks
            if total_tokens > max_input_tokens:
                print(f"Text is too large for a single API call (est. {total_tokens} tokens > {max_input_tokens}). Processing in chunks...")
                
                # Calculate optimal chunk size based on token count
                # We want to maximize chunk size while staying under the token limit
                avg_page_tokens = (native_tokens + ocr_tokens) / total_pages
                # Allow for prompt tokens and some buffer
                tokens_per_chunk = max_input_tokens - prompt_tokens - 500  # 500 token buffer
                optimal_chunk_size = max(1, int(tokens_per_chunk / avg_page_tokens))
                
                print(f"Calculated optimal chunk size: {optimal_chunk_size} pages (avg {avg_page_tokens:.1f} tokens per page)")
                return self._process_in_chunks(client, native_txt_pages, ocr_txt_pages, images, out_dir, optimal_chunk_size)
            else:
                print(f"Processing entire document in a single API call (est. {total_tokens} tokens)")
                # Create a detailed prompt for the AI
                prompt = self.prompt_template.format(
                    chunk_info="",
                    native_text=full_native_text,
                    ocr_text=full_ocr_text
                )
                
                # Track the token count of the actual prompt
                actual_prompt_tokens = self._count_tokens(prompt)                
                print(f"Actual prompt token count: {actual_prompt_tokens}")
                
                # Save the prompt to a file for debugging
                prompt_filename = "prompt-full-document.txt"
                prompt_filepath = os.path.join(out_dir, prompt_filename)
                with open(prompt_filepath, 'w', encoding='utf-8') as f:
                    f.write(prompt)
                print(f"Saved prompt to file: {prompt_filepath}")

            print(f"Sending {len(prompt)} characters to the AI for processing...")
            # Call OpenAI API with retry logic
            prompt_filename = "prompt-full-document.txt"
            cleaned_text = self._call_api_with_retry(client, prompt, out_dir, prompt_filename, full_native_text)
            
            # Add image references
            if all_image_refs:
                cleaned_text += "\n\n" + "\n\n".join(all_image_refs)
            
            # Write the AI-friendly version to file
            ai_file = os.path.join(out_dir, "exported-ai.txt")
            with open(ai_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"Created AI-friendly text file: {ai_file}")
            return ai_file
            
        except Exception as e:
            print(f"Error creating AI-friendly version: {e}")
            
            # Attempt to fallback to a different model if there's an API error
            if "model" in str(e).lower() or "parameter" in str(e).lower():
                fallback_model = "gpt-3.5-turbo" if self.model != "gpt-3.5-turbo" else "gpt-4"
                print(f"\nAttempting fallback to {fallback_model}...")
                try:
                    # Save original model name
                    original_model = self.model
                    # Set fallback model
                    self.model = fallback_model
                    # Update model type flags
                    self.is_claude_model = self.model.startswith('claude-') or self.model.startswith('anthropic/')
                    self.is_o_series = self.model.startswith('o') and not self.model.startswith('openai/')
                    self.is_gpt_model = 'gpt' in self.model.lower()
                    # Try again with fallback model
                    return self.create_ai_friendly_version(native_txt_pages, ocr_txt_pages, images, out_dir)
                except Exception as fallback_e:
                    print(f"Fallback to {fallback_model} also failed: {fallback_e}")
                    # Restore original model name
                    self.model = original_model
                    self.is_claude_model = self.model.startswith('claude-') or self.model.startswith('anthropic/')
                    self.is_o_series = self.model.startswith('o') and not self.model.startswith('openai/')
                    self.is_gpt_model = 'gpt' in self.model.lower()
            return None
    
    def _process_in_chunks(self, client, native_txt_pages, ocr_txt_pages, images, out_dir, chunk_size):
        """
        Process the document in chunks when it's too large for a single API call.
        
        Args:
            client: OpenAI client
            native_txt_pages: List of text pages from native extraction
            ocr_txt_pages: List of text pages from OCR extraction
            images: Dictionary mapping page numbers to lists of image filenames
            out_dir: Output directory for the AI-friendly version
            chunk_size: Number of pages per chunk
            
        Returns:
            str: Path to the created AI-friendly file, or None if failed
        """
        total_pages = min(len(native_txt_pages), len(ocr_txt_pages))
        chunks = []
        
        # Initialize token tracking
        total_tokens_used = 0
        completion_tokens_used = 0
        
        # Define token limits based on the model
        if self.is_o_series:
            max_input_tokens = 128000  # o4 models have large context windows
        elif self.is_gpt_model and "gpt-4" in self.model:
            max_input_tokens = 8000  # GPT-4 has a larger context window
        else:
            max_input_tokens = 6000  # Conservative limit for most models
        
        # Process document in chunks
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            print(f"Processing chunk: pages {chunk_start+1}-{chunk_end} of {total_pages}")
            
            # Prepare text for this chunk
            chunk_native = []
            chunk_ocr = []
            for i in range(chunk_start, chunk_end):
                chunk_native.append(f"--- Page {i+1} ---\n{native_txt_pages[i]}\n--- End of Page {i+1} ---")
                chunk_ocr.append(f"--- Page {i+1} ---\n{ocr_txt_pages[i]}\n--- End of Page {i+1} ---")
            
            full_native_chunk = "\n\n".join(chunk_native)
            full_ocr_chunk = "\n\n".join(chunk_ocr)
            
            # Create a detailed prompt for the AI chunk
            prompt = self.prompt_template.format(
                chunk_info=f"You are currently processing a CHUNK of the document (pages {chunk_start+1}-{chunk_end}).",
                native_text=full_native_chunk,
                ocr_text=full_ocr_chunk
            )
            
            # Calculate token usage for this chunk
            chunk_prompt_tokens = self._count_tokens(prompt)
            print(f"Chunk {chunk_start+1}-{chunk_end} prompt token count: {chunk_prompt_tokens}")
            
            prompt_filename = f"prompt-chunk-{chunk_start+1}-{chunk_end}.txt"
            chunk_text = self._call_api_with_retry(client, prompt, out_dir, prompt_filename, full_native_chunk)
            
            # Track token usage if response has usage information
            try:
                # This assumes the _call_api_with_retry method returns the response object or has access to it
                if hasattr(client, 'last_response') and hasattr(client.last_response, 'usage'):
                    usage = client.last_response.usage
                    total_tokens_used += usage.total_tokens
                    completion_tokens_used += usage.completion_tokens
                    print(f"Chunk {chunk_start+1}-{chunk_end} token usage: {usage.total_tokens} total, {usage.completion_tokens} completion")
                    
                    # Dynamically adjust chunk size for next iteration
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
            except Exception as e:
                print(f"Error tracking token usage: {e}")
            
            # Write the individual chunk to its own file for debugging
            chunk_filename = f"exported-ai-chunk-{chunk_start+1}-{chunk_end}.txt"
            chunk_filepath = os.path.join(out_dir, chunk_filename)
            with open(chunk_filepath, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
            print(f"Created AI chunk file: {chunk_filepath}")
            
            chunks.append(chunk_text)
        
        # Combine all chunks
        combined_text = "\n\n".join(chunks)
        
        # Add image references
        image_refs = []
        for page_num, page_images in images.items():
            if page_images:
                refs = [f"[IMAGE: {img}]" for img in page_images]
                image_refs.append(f"--- Images on page {page_num} ---\n" + "\n".join(refs) + "\n--- End of images ---")
        
        if image_refs:
            combined_text += "\n\n" + "\n\n".join(image_refs)
        
        # Print token usage summary if available
        if total_tokens_used > 0:
            print("\nToken usage summary:")
            print(f"Total tokens used: {total_tokens_used}")
            print(f"Completion tokens used: {completion_tokens_used}")
            print(f"Prompt tokens used: {total_tokens_used - completion_tokens_used}")
            
            # Estimate cost (very rough approximation)
            prompt_tokens_cost = 0
            completion_tokens_cost = 0
            
            if "gpt-4" in self.model:
                prompt_tokens_cost = 0.00003 * (total_tokens_used - completion_tokens_used)
                completion_tokens_cost = 0.00006 * completion_tokens_used
            elif "gpt-3.5" in self.model:
                prompt_tokens_cost = 0.0000015 * (total_tokens_used - completion_tokens_used)
                completion_tokens_cost = 0.000002 * completion_tokens_used
            elif self.is_o_series:
                prompt_tokens_cost = 0.000005 * (total_tokens_used - completion_tokens_used)
                completion_tokens_cost = 0.000015 * completion_tokens_used
            else:
                prompt_tokens_cost = 0.00001 * (total_tokens_used - completion_tokens_used)
                completion_tokens_cost = 0.00002 * completion_tokens_used
                
            total_cost = prompt_tokens_cost + completion_tokens_cost
            print(f"Estimated cost: ${total_cost:.6f} (input: ${prompt_tokens_cost:.6f}, output: ${completion_tokens_cost:.6f})")
        
        # Write the AI-friendly version to file
        ai_file = os.path.join(out_dir, "exported-ai.txt")
        with open(ai_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"Created AI-friendly text file: {ai_file}")
        return ai_file