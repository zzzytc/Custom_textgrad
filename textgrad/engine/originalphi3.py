import os
import platformdirs
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests

from .base import EngineLM, CachedEngine
from PIL import Image


class originalChatphi3(EngineLM, CachedEngine):
    def __init__(
            self, 
            model_name = "microsoft/Phi-3-vision-128k-instruct", 
            processor_name = "microsoft/Phi-3-vision-128k-instruct",
            max_seq_length=2048, 
            dtype='auto', 
            load_in_4bit=True,
            HF_TOKEN=None):
        
        self.model_name = model_name
        self.processor_name = processor_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.HF_TOKEN = HF_TOKEN
        # Initialize paths and processor
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_together_{model_name}.db")
        
        super().__init__(cache_path=cache_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

        self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
            
    def generate(self, question, image_path=None, system_prompt = None, eos_token_id=None ):
        
        eos_token_id = self.processor.tokenizer.eos_token_id

        messages = [ 
            {"role": "user", "content": f"<|image_1|>\n{system_prompt}"}, 
            {"role": "user", "content": f"{question}"} ]
        if image_path != None:    
            image = Image.open(requests.get(image_path, stream=True).raw)
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")
        
        print(f"#####################DEBUG_prompt: {prompt}")
        if image_path != None: 
            inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        else:
            inputs = self.processor(prompt, images=None, return_tensors="pt").to("cuda:0")

        generation_args = { "max_new_tokens": 500, "temperature": 0.0, "do_sample": False }
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        print (f"################DEBUG response{response}")
        return response

            


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)