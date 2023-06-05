# pips some might be unneccesary
# numba
# cuda-python
# transformers accelerate einops langchain xformers
# pip from pytorch website not working, adding --upgrade --force-restart helps get the correct version. long downlaod...
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall

# an attempt to learn how to implement a local LLM using Mosaicml MPT-7b
# going with the "chat" version 
# it is also partially trained on Anthropic HH_RLHF!


# from langchain.llms import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else'cpu'
print(torch.__version__)
print(f"Model loaded on {device}")

def init_model(device):
    print("Loading model... ")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b-chat',
        trust_remote_code=True,
        torch_dtype=bfloat16,
        max_seq_len=2048
    )
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")
    return model


# initalize tokenizer
def init_tokenizer():
    print("Init tokenizer..." )
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    return tokenizer


# # mtp-7b is trianged to add "</endoftext/>" at the end of generations
def get_stopping_criteria(tokenizer):
    print("Defininf stopping criteria... ")
    class StopOnTokens(StoppingCriteria):
        def __init__(self, tokenizer):
            self.stop_token_ids = tokenizer.convert_tokens_to_ids([""])

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            return any(input_ids[0][-1] == stop_id for stop_id in self.stop_token_ids)
    return StoppingCriteriaList([StopOnTokens(tokenizer)])
    
        

# initalize the HF pipeline
def init_hf_pipeline(model, tokenizer, device, stopping_criteria):
    print("Initalizing Hugging Face pipeline... ")
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        device=device,
        stopping_criteria=stopping_criteria,
        temperature=0.1,
        top_p=0.15,
        top_k=0,
        max_new_tokens=64, # not sure if this is what's causing short responses or the stop criteria is too narrow
        repetition_penalty=1.1
    )
    return generate_text

# confirmation this works
model = init_model(device)
tokenizer = init_tokenizer()
stopping_criteria = get_stopping_criteria(tokenizer)
generate_text = init_hf_pipeline(model, tokenizer, device, stopping_criteria)

res = generate_text("Write a sea shanty about eating a sandwich on the open seas")
print(res[0]["generated_text"])