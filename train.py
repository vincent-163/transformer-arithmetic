import main
import pytorch_lightning as pl

if __name__ == "__main__":
    m = Model()
    trainer = pl.Trainer(
        gpus=4,
        accumulate_grad_batches=8,
        checkpoint_callback=ModelCheckpoint(),
        distributed_backend="ddp")
    trainer.fit(m)