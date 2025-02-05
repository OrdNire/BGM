from torch.utils.data import Dataset, DataLoader
import torch
import random

class RSICD_finetune(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encodings = []
        for caption in item["captions"]:
            encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["text"] = caption["raw"]
            encodings.append(encoding)
        return encodings

        # encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # # remove batch dimension
        # encoding = {k: v.squeeze() for k, v in encoding.items()}
        # encoding["text"] = random.choice(item["captions"])["raw"]
        # return encoding

def collate_fn(batch, processor):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0][0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([encoding[key] for example in batch for encoding in example])
        else:
            text_inputs = processor.tokenizer(
                [encoding["text"] for example in batch for encoding in example], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch