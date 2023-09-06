from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import csv

################################
#    With Generate Function    #
#  USE watch -n 0.1 nvidia-smi #
################################

# model_name = "gpt2" # 548M 487MB
# model_name = "gpt2-medium" # 1.52GB 1377MB
# model_name = "gpt2-large" # 3.52GB 3061MB
# model_name = "gpt2-xl" # 6.43GB 6124MB

# model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
name = "gpt2"
model_names = [name]
batches = [1, 2, 4, 8, 16]
# batches = [8, 16, 32, 64, 128, 256, 512]
output_lens = [96, 224, 480, 992]
# output_lens = [480]
device_int = 0
device = 'cuda:' + str(device_int)

with open('result/'+name+'-length.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['total length', 'memory'])

    for model_name in model_names:

        # torch.cuda.reset_peak_memory_stats(device=device_int)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
        print("Model: " + model_name, end=" ")
        print(int(torch.cuda.max_memory_allocated(device=device_int)/1024/1024), end="")
        print("MB")

        for batch in batches:
            for output_len in output_lens: # 993 (1024 + 1 - 32)

                input = tokenizer([("The " * 31 + "end" ) for _ in range(batch)], return_tensors="pt").to(device)

                # print(input["input_ids"].device)

                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                #     with record_function("model_inference"):

                torch.cuda.reset_peak_memory_stats(device=device_int)
                output = model.generate(**input, max_length=output_len + 32)
                mem = torch.cuda.max_memory_allocated(device=device_int)
                print("batch " + str(batch) + " length " + str(output_len + 32) + ": ", end="")
                print(int(mem/1024/1024), end="")
                print("MB")

                wr.writerow([output_len + 32, mem])

                # print(prof.key_averages().table(row_limit=10))
                # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

                # print(tokenizer.decode(output[0], skip_special_tokens=True))


# make graph
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('result/'+name+'-length.csv', header=None)
plt.plot(df)
df.plot(x=0, y=1, xlabel='total length', ylabel='Bytes', title=name+" memory")
# plt.show()
plt.savefig('result/'+name+'-length.png')