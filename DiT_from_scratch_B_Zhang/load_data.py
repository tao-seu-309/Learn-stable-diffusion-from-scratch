from datasets import load_dataset

ds = load_dataset("svjack/pokemon-blip-captions-en-zh")
ds.save_to_disk("./data")
print("---")
print("success")
