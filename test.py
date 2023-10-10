from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

model_path = "./Llama-2-13B-chat-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(
                                            model_path,
                                            device_map="auto",
                                            trust_remote_code=True,
                                            revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

memory = """ [INST] <<SYS>>
    You are Paul Graham co-founder of the start up accelerator Y Combinator. Answer any question as Paul Graham
    <<SYS>>
"""
try:
    while True:
        user_input = input("You: ")
        history = memory + "\n" + user_input + tokenizer.eos_token
        response = pipe(history)
        generated_text = response[0]["generated_text"]
        memory = history + generated_text

        print("Paul Graham: ", generated_text)
except:
    print(memory)
