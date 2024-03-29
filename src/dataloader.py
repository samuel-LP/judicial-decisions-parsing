import torch


class CustomCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = \
            [sample["attention_mask"] for sample in batch]
        output["sexe"] = [sample["sexe"] for sample in batch]
        output["text_id"] = [sample["text_id"] for sample in batch]

        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = \
                [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]

            output["attention_mask"] = \
                [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]

        else:

            output["input_ids"] = \
                [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = \
                [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"],
                                           dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"],
                                                dtype=torch.long)

        sexe_to_int = {"homme": 0,
                       "femme": 1,
                       "n.c.": -1}
        output["sexe"] = \
            torch.tensor([sexe_to_int[item] for item in output["sexe"]],
                         dtype=torch.long)
        output["text_id"] = torch.tensor(output["text_id"], dtype=torch.long)
        return output
