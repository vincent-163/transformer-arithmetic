import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import expr
import math
from pytorch_lightning.callbacks import ModelCheckpoint

class Model(pl.LightningModule):
    def __init__(self, batch_size=16):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.model = model.GPT2LMHeadModel(
            vocab_size=128,
            n_positions=1024,
            n_layer=4,
            n_embd=512,
            n_ctx=1024,
            n_state=2048,
            n_head=8,
            dropout=0.9
        )

    def forward(self, x):
        return self.model(x)
    
    def get_losses(self, strs, begins=None):
        lengths = [len(s) for s in strs]
        length = max(lengths)
        strs = torch.stack([torch.tensor(bytearray(s.ljust(length, '\0'), 'ascii'), device=self.device) for s in strs])
        logits, _ = self.model.forward(strs)
        result = []
        for i, (s, l, v) in enumerate(zip(strs, lengths, logits)):
            if begins is None:
                begin = 0
            else:
                begin = begins[i]
            result.append(F.cross_entropy(v[begin:l-1], s[begin+1:l], reduction='sum'))
        return result
    
    def train_dataloader(self):
        dataset = expr.ExpressionDataset(100000)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    def training_step(self, batch, batch_nb):
        losses = self.get_losses(batch)
        loss = torch.stack(losses).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.00025)
    
    def validation_step(self, batch, batch_nb):
        losses = self.get_losses(batch, begins=[len("12345*54321;")]*len(batch))
        losses = torch.stack(losses)
        return {'val_loss': losses.mean(), 'expected_accuracy': torch.exp(-losses).mean()}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['expected_accuracy'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss, 'avg_accuracy': avg_accuracy}}
    
    def val_dataloader(self):
        dataset = expr.ExpressionDataset(1000, 100000)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def test_step(self, batch, batch_idx):
        correct, total = 0., 0.
        for (a, b), y in batch:
            prompt = "%d*%d;" % (a, b)
            result = self.generate(prompt, 1024)
            answer = expr.get_multiplication_answer(result)
            total += 1
            if answer == y:
                correct += 1
        return {'accuracy': correct / total, 'correct': correct, 'total': total}
    
    def test_eopch_end(self, outputs):
        correct = torch.stack([x['correct'] for x in outputs]).sum()
        total = torch.stack([x['total'] for x in outputs]).sum()
        return {'accuracy': correct / total, 'correct': correct, 'total': total}
        
    def test_dataloader(self):
        dataset = expr.TestDataset(1000, 200000)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader
    
    @torch.no_grad()
    def generate(self, expr, max_length):
        self.eval()
        curi = 0
        past = None
        while len(expr) < max_length:
            cur = torch.tensor([bytearray(expr[curi:], 'ascii')], dtype=torch.long, device=self.device)
            probs, past = self.model(cur, past=past)
            curi = len(expr)
            probs = F.softmax(probs[0, -1], dim=-1)
            sample = torch.multinomial(probs, 1)[0]
            expr += chr(sample)
            if sample == ord('$'):
                break
        return expr
