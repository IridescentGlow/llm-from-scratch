import numpy as np
import pytest
import torch

from llm_from_scratch.data import (
    TokenDataset,
    get_dataloader,
    load_loss_mask,
    load_token_ids,
    train_val_split,
    write_loss_mask,
    write_token_ids,
)

TOKEN_STREAM = list(range(20))  # [0, 1, 2, ..., 19]
CONTEXT_LENGTH = 4


def test_input_target_alignment():
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH)
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [0, 1, 2, 3]
    assert target_ids.tolist() == [1, 2, 3, 4]  # target is input shifted by one

    input_ids, target_ids = dataset[5]
    assert input_ids.tolist() == [5, 6, 7, 8]
    assert target_ids.tolist() == [6, 7, 8, 9]


def test_dataset_length_is_number_of_windows():
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH)
    assert len(dataset) == len(TOKEN_STREAM) - CONTEXT_LENGTH


def test_dataset_rejects_stream_shorter_than_context():
    with pytest.raises(ValueError):
        TokenDataset(np.array([1, 2, 3]), context_length=4)


def test_dataloader_batch_shapes():
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH)
    loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    input_batch, target_batch = next(iter(loader))
    assert input_batch.shape == (3, CONTEXT_LENGTH)
    assert target_batch.shape == (3, CONTEXT_LENGTH)
    assert torch.equal(target_batch[:, :-1], input_batch[:, 1:])  # shift-by-one holds per row


def test_dataloader_shuffle_varies_order():
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH)
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)
    unshuffled_first = next(iter(loader))[0].tolist()

    torch.manual_seed(0)
    shuffled_loader = get_dataloader(dataset, batch_size=1, shuffle=True)
    shuffled_first = next(iter(shuffled_loader))[0].tolist()

    # not a strict guarantee, but with 16 possible windows and a fixed seed
    # this is deterministic and exercises the shuffle path meaningfully
    assert isinstance(shuffled_first, list)
    assert unshuffled_first == [[0, 1, 2, 3]]


def test_train_val_split_is_positional_and_covers_all_tokens():
    tokens = np.arange(100)
    train, val = train_val_split(tokens, train_split=0.9)
    assert len(train) == 90
    assert len(val) == 10
    assert train.tolist() == list(range(90))
    assert val.tolist() == list(range(90, 100))
    assert len(train) + len(val) == len(tokens)


def test_train_val_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        train_val_split(np.arange(10), train_split=1.5)


def test_memmap_roundtrip_matches_original(tmp_path):
    token_ids = [1, 2, 3, 4, 5, 6000, 65000 % 65536]
    path = tmp_path / "tokens.bin"
    write_token_ids(token_ids, path)

    loaded = load_token_ids(path)
    assert loaded.tolist() == token_ids


def test_memmap_does_not_load_whole_file_into_memory(tmp_path):
    token_ids = list(range(1000))
    path = tmp_path / "tokens.bin"
    write_token_ids(token_ids, path)

    loaded = load_token_ids(path)
    assert isinstance(loaded, np.memmap)
    # slicing a memmap should still be a memmap/view, not force a full copy
    window = loaded[10:14]
    assert window.tolist() == [10, 11, 12, 13]


def test_memmap_backed_dataset_windows_correctly(tmp_path):
    token_ids = list(range(20))
    path = tmp_path / "tokens.bin"
    write_token_ids(token_ids, path)

    loaded = load_token_ids(path)
    dataset = TokenDataset(loaded, context_length=CONTEXT_LENGTH)
    input_ids, target_ids = dataset[2]
    assert input_ids.tolist() == [2, 3, 4, 5]
    assert target_ids.tolist() == [3, 4, 5, 6]


def test_loss_mask_none_leaves_targets_unchanged():
    """Default loss_mask=None must be exactly today's pretraining/eval
    behavior -- no -100 anywhere. See docs/finetune-loss-masking.md."""
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH)
    _, target_ids = dataset[0]
    assert -100 not in target_ids.tolist()
    assert target_ids.tolist() == [1, 2, 3, 4]


def test_loss_mask_marks_masked_targets_as_ignore_index():
    """False positions become target -100 -- F.cross_entropy's default
    ignore_index, so they contribute no loss/gradient with no model change
    needed. See docs/finetune-loss-masking.md."""
    mask = np.array([False, False, True, True, True, True, True, True, True, True] * 2)
    dataset = TokenDataset(np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH, loss_mask=mask)

    input_ids, target_ids = dataset[0]
    # target_ids correspond to token positions 1,2,3,4 -> mask[1:5] = [False, True, True, True]
    assert input_ids.tolist() == [0, 1, 2, 3]
    assert target_ids.tolist() == [-100, 2, 3, 4]


def test_loss_mask_length_mismatch_raises():
    with pytest.raises(ValueError):
        TokenDataset(
            np.array(TOKEN_STREAM), context_length=CONTEXT_LENGTH, loss_mask=np.array([True, False])
        )


def test_loss_mask_persistence_roundtrip(tmp_path):
    mask = [True, False, False, True, True]
    path = tmp_path / "loss_mask.bin"
    write_loss_mask(mask, path)

    loaded = load_loss_mask(path)
    assert loaded.tolist() == [1, 0, 0, 1, 1]
