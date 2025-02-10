import os
import base64
from typing import List, Union, Dict
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

class QwenVLProcessor:
    def __init__(  
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        use_flash_attention: bool = True,
        min_pixels: int = 128*16*16,
        max_pixels: int = 1024*16*16
    ):
        """
        Initialize the QwenVL processor with custom configuration.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to run the model on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention
            min_pixels: Minimum number of pixels for image processing
            max_pixels: Maximum number of pixels for image processing
        """
        # Configure CUDA memory allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Clear CUDA cache
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Load model and assign to self
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            use_flash_attention_2=use_flash_attention,
            use_cache=True
        )
        
        # Load processor and assign to self
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        self.device = device

    def _encode_image(self, image_path: str) -> str:
        """
        Encode a local image file to base64.
        
        Args:
            image_path: Path to the local image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

    def prepare_messages(
        self,
        image_paths: Union[str, List[str]],
        prompt: str
    ) -> List[Dict]:
        """
        Prepare messages for the model using local image paths.
        
        Args:
            image_paths: Single path or list of paths to local images
            prompt: Text prompt to process with the images
            
        Returns:
            List of formatted messages for the model
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        messages = []
        for path in image_paths:
            encoded_image = self._encode_image(path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": encoded_image},
                    {"type": "text", "text": prompt}
                ]
            })
        return messages

    def process_images(
        self,
        image_paths: Union[str, List[str]],
        prompt: str,
        max_new_tokens: int = 60000,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Process local images with the given prompt.
        
        Args:
            image_paths: Single path or list of paths to local images
            prompt: Text prompt to process with the images
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated responses for each image
        """
        messages = self.prepare_messages(image_paths, prompt)
        
        with torch.inference_mode():
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(self.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
        return output_text

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = QwenVLProcessor()
    
    # Process single image
    image_path = "./image.jpeg"
    result = processor.process_images(
        image_path,
        prompt="""You are an expert OCR model who can read and interpret hard images in details 
        and in great precision. Given these images extract every detail of it in an organized format."""
    )
    print(f"Single image result: {result[0]}")
