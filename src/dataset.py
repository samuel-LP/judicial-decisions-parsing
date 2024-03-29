from torch.utils.data import Dataset


class JuridiqueDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 args):
        # args is a dict, a nice way to share the global arguments
        # (even accross multiple files)
        self.args = args
        self.tokenizer = tokenizer
        self.df = df

    def make_one_item(self, idx):
        # this function should encode (tokenize) a given text
        text_id = self.df.iloc[idx].text_id
        text = self.df.iloc[idx].texte
        sexe = self.df.iloc[idx].sexe
        tokenizer_encoding = self.tokenizer(text, max_length=512)
        outputs = dict(**tokenizer_encoding)

        outputs['text_id'] = text_id
        outputs['sexe'] = sexe

        return outputs

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return self.make_one_item(idx)
