import os
import platformdirs
from tenacity import retry, stop_after_attempt, wait_random_exponential
from unsloth import FastLanguageModel
from transformers import pipeline, TextStreamer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from .base import EngineLM, CachedEngine

class ChatLlama(EngineLM, CachedEngine):
    # DEFAULT_SYSTEM_PROMPT = STARTING_SYSTEM_PROMPT

    def __init__(
        self,
        model_string="unsloth/llama-3-8b-Instruct-bnb-4bit",
        system_prompt=None,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    ):
        """
        :param model_string: The model identifier string.
        :param system_prompt: The default system prompt.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_together_{model_string}.db")
        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.model_string = model_string

        # Load model with Unsloth in 4 bits
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_string,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Init for inference
        FastLanguageModel.for_inference(self.model)

        # Init TextStreamer
        self.streamer = TextStreamer(self.tokenizer)

        # Init pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            streamer=self.streamer,
            return_full_text=False
        )

        # Put HF pipeline into llm for inference
        self.llm = HuggingFacePipeline(pipeline=self.pipe)


    def generate(
            self, prompt, system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
        ):
            sys_prompt_arg = str(system_prompt) if system_prompt else self.system_prompt

            cache_or_none = self._check_cache(sys_prompt_arg + str(prompt))
            if cache_or_none is not None:
                return cache_or_none

            # Combine system prompt and user prompt with tags
            combined_prompt = f"{sys_prompt_arg}\n{str(prompt)}"

            print("Generating response...")
            try:
                # Generate results
                res = self.llm(combined_prompt)
                print(f"############################res {res}")
                response = res[0]['generated_text'] if "generated_text" in res[0] else res[0]
                print(f"############################responaw {response}")
            except AttributeError as e:
                print(f"Error during generation: {e}")
                raise e

            self._save_cache(sys_prompt_arg + str(prompt), response)
            print(f"############################response {response}")
            return res

   
    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

