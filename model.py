import time
import openai
# from openai import OpenAI
import openai
from transformers import AutoTokenizer
import torch
import transformers
from prompts import identity
import pdb
import backoff
from pprint import pprint



def get_model(args):
    model_name, temperature = args.model, args.temperature
    if 'gpt' in model_name:
        model = GPT(args.api_key, model_name, temperature)
        return model
    elif 'llama' in model_name:
        return LLaMA(model_name, temperature)


class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, api_key, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        openai.api_key = api_key

    # @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def get_response(self, **kwargs):
        # return self.client.chat.completions.create(**kwargs)
        try:
            res = openai.ChatCompletion.create(**kwargs)
            return res
        except openai.APIConnectionError as e:
            print(f'APIConnectionError: {e}')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.RateLimitError as e:
            print(f'RateLimitError: {e}')
            exit()
            return self.get_response(**kwargs)
        except openai.APITimeoutError as e:
            print(f'APITimeoutError: {e}')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.BadRequestError as e:
            raise e

    def forward(self, head, prompts):
        messages = [
            {"role": "system", "content": head}
        ]
        info = {}
        for i, prompt in enumerate(prompts):
            messages.append(
                {"role": "user", "content": prompt}
            )
            try:
                response = self.get_response(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
            except openai.BadRequestError as e:
                raise e
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            info = dict(response.usage)  # completion_tokens, prompt_tokens, total_tokens
            info['response'] = messages[-1]["content"]
            info['message'] = messages
        return self.post_process_fn(info['response']), info


class LLaMA(Model):
    def __init__(self, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer,
            temperature=temperature
        )

    def forward(self, head, prompts):
        prompt = prompts[0]
        sequences = self.pipeline(
            prompt,
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = sequences[0]['generated_text']  # str
        info = {
            'message': prompt,
            'response': response
        }
        return self.post_process_fn(info['response']), info
