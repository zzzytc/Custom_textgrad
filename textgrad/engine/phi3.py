import os
import platformdirs
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests

from .base import EngineLM, CachedEngine
from PIL import Image


class Chatphi3(EngineLM, CachedEngine):
    def __init__(
            self, 
            model_name = "catyung/phi-3-vision-128k-the-wave", 
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


        # # Load the model with BitsAndBytes configuration
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=load_in_4bit,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="bfloat16",
        #     bnb_4bit_use_double_quant=True,
        # )
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=dtype,
        #     trust_remote_code=True,
        #     quantization_config=self.bnb_config,
        #     device_map='auto',
        #     attn_implementation='flash_attention_2',
        #     token= HF_TOKEN
        # )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

        self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        
        # Initialize the pipeline
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     processor=self.processor,
        #     max_new_tokens=20,
        #     temperature=0.0,
        #     do_sample=False,
        #     return_full_text=False
        # )
    
    def generate(self, question, image_path, system_prompt = None, eos_token_id=None, generation_args = { "max_new_tokens": 500, "temperature": 0.0, "do_sample": False } ):
        eos_token_id = self.processor.tokenizer.eos_token_id

        messages = [ 
            {"role": "user", "content": f"<|image_1|>\n{system_prompt}"}, 
            {"role": "user", "content": f"{question}"} ]
            
        image = Image.open(requests.get(image_path, stream=True).raw)
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")

        print(f"#####################DEBUG_prompt: {prompt}")

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        # print("Generating response...")
        # try:
        #     response = self.pipeline(prompt, max_length=max_tokens, return_full_text=False)
        #     print("Generated response:", response)
        # except Exception as e:
        #     print(f"Error during generation: {e}")
        #     raise

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        print (f"################DEBUG response{response}")
        return response
    
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)