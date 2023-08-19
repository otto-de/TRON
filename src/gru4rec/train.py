from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

from src.gru4rec.dataset import Gru4RecDataset
from src.gru4rec.model import GRU4REC


def train_gru(config, data_dir, train_stats, test_stats, num_items):
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                          monitor='recall_cutoff_20',
                                          mode='max',
                                          filename=f'gru4rec-{config["dataset"]}-' + '{epoch}-{recall_cutoff_20:.3f}')

    trainer = Trainer(max_epochs=config["max_epochs"],
                      precision=16,
                      limit_val_batches=config["limit_val_batches"],
                      log_every_n_steps=1,
                      accelerator=config["accelerator"],
                      devices=1,
                      overfit_batches=config["overfit_batches"],
                      callbacks=[checkpoint_callback])

    train_set = Gru4RecDataset(f'{data_dir}/{config["dataset"]}/{config["dataset"]}_train.jsonl',
                               train_stats["num_sessions"],
                               num_items=num_items,
                               max_seqlen=config["max_session_length"],
                               shuffling_style=config["shuffling_style"],
                               num_in_batch_negatives=config["num_batch_negatives"],
                               num_uniform_negatives=config["num_uniform_negatives"],
                               reject_uniform_session_items=config["reject_uniform_session_items"],
                               reject_in_batch_items=config["reject_in_batch_items"],
                               sampling_style=config["sampling_style"],
                               batch_size=config["batch_size"])

    test_set = Gru4RecDataset(f'{data_dir}/{config["dataset"]}/{config["dataset"]}_test.jsonl',
                              test_stats["num_sessions"],
                              num_items=num_items,
                              max_seqlen=config["max_session_length"],
                              shuffling_style="no_shuffling",
                              num_in_batch_negatives=config["num_batch_negatives"],
                              num_uniform_negatives=config["num_uniform_negatives"],
                              reject_uniform_session_items=config["reject_uniform_session_items"],
                              reject_in_batch_items=config["reject_in_batch_items"],
                              sampling_style=config["sampling_style"],
                              batch_size=config["batch_size"])

    train_loader = DataLoader(
        train_set,
        drop_last=True,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=train_set.dynamic_collate,
        prefetch_factor=100)

    test_loader = DataLoader(
        test_set,
        drop_last=True,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        collate_fn=test_set.dynamic_collate,
        prefetch_factor=10)

    model = GRU4REC(hidden_size=config["hidden_size"],
                    dropout_rate=config["dropout"],
                    num_items=num_items,
                    learning_rate=config["lr"],
                    batch_size=config["batch_size"],
                    sampling_style=config["sampling_style"],
                    topk_sampling=config.get("topk_sampling", False),
                    topk_sampling_k=config.get("topk_sampling_k", 1000),
                    num_layers=config["num_layers"],
                    loss=config["loss"],
                    bpr_penalty=config.get("bpr_penalty", None),
                    optimizer=config["optimizer"],
                    output_bias=config["output_bias"],
                    share_embeddings=config["share_embeddings"],
                    original_gru=config["original_gru"],
                    final_activation=config["final_activation"])

    return trainer, model, train_loader, test_loader
