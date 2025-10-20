import os, torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig, BertForMaskedLM,
    TrainingArguments, Trainer,
)
import numpy as np

def mlm_accuracy(eval_pred):
    logits, labels = eval_pred
    # logits: (N, T, V); labels: (N, T) with -100 where not masked
    preds = logits.argmax(-1)
    mask = labels != -100
    if mask.sum() == 0:
        return {"mlm_acc": 0.0}
    correct = (preds[mask] == labels[mask]).sum()
    total = mask.sum()
    return {"mlm_acc": (correct / total).item()}


def load_batch_and_pad(t, L, pad_val):
    if t.size(1) == L:    # already the right length
        return t
    pad = t.new_full((t.size(0), L - t.size(1)), pad_val)
    return torch.cat([t, pad], dim=1)

def load_split(path, pad_id, label_pad=-100):
    """
    Each .pt file is a list of batch dicts with tensors:
      {input_ids: (B, T_b), attention_mask: (B, T_b), labels: (B, T_b)}
    We re-pad all batches to the same max length, then concatenate along batch dim.
    """
    batches = torch.load(path, map_location="cpu")
    if not batches:
        raise ValueError(f"No batches found in {path}")

    max_len = max(b["input_ids"].size(1) for b in batches)
    outs = {"input_ids": [], "attention_mask": [], "labels": []}

    for b in batches:
        outs["input_ids"].append(load_batch_and_pad(b["input_ids"], max_len, pad_id))
        outs["attention_mask"].append(load_batch_and_pad(b["attention_mask"], max_len, 0))
        outs["labels"].append(load_batch_and_pad(b["labels"], max_len, label_pad))

    input_ids = torch.cat(outs["input_ids"], dim=0)
    attention_mask = torch.cat(outs["attention_mask"], dim=0)
    labels = torch.cat(outs["labels"], dim=0)

    class MLMDataset(Dataset):
        def __len__(self): return input_ids.size(0)
        def __getitem__(self, i):
            return {
                "input_ids": input_ids[i],
                "attention_mask": attention_mask[i],
                "labels": labels[i],
            }
    return MLMDataset()


if __name__ == "__main__":
    DATA_DIR = "mlm_data"                      # where train.pt / validation.pt / test.pt live
    TOKENIZER_JSON = "python_tokenizer.json"   
    MAX_POS_EMBED = 512        
    
    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_JSON)
    tok.add_special_tokens({
        "pad_token": "<pad>", "unk_token": "<unk>", "mask_token": "<mask>",
        "bos_token": "<s>", "eos_token": "</s>",
    })
    vocab_size = len(tok)          
    
    print(f"Loading datasets from {DATA_DIR}...")
    train_ds = load_split(os.path.join(DATA_DIR, "train.pt"), pad_id=tok.pad_token_id)
    val_ds   = load_split(os.path.join(DATA_DIR, "validation.pt"), pad_id=tok.pad_token_id)
    test_ds  = load_split(os.path.join(DATA_DIR, "test.pt"), pad_id=tok.pad_token_id) 
    print(f"Done dataset load.")
    
    
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,                 
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=max(MAX_POS_EMBED, 512),
        pad_token_id=tok.pad_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertForMaskedLM(config)
    
    args = TrainingArguments(
        output_dir="mlm_model_bert",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        learning_rate=5e-4,
        num_train_epochs=3,  # TODO: increase this after debugging
        logging_steps=50_000,
        save_steps=50_000,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        compute_metrics=mlm_accuracy,
    )
            
    trainer.train()
    
    trainer.save_model("mlm_model_bert")   # saves config + weights
    tok.save_pretrained("mlm_model_bert")  # saves tokenizer files next to model
    print("✅ Saved model + tokenizer to mlm_model_bert")
    
    # Final evaluation on test set
    with torch.no_grad():
        list_test_acc = []
        for test_batch in test_ds:
            test_batch = {k: v.unsqueeze(0).to(trainer.args.device) for k, v in test_batch.items()}
            outputs = model(**test_batch)
            
            logits = outputs.logits
            preds = logits.argmax(-1)
        
            acc = mlm_accuracy((logits.cpu().numpy(), test_batch['labels'].cpu().numpy()))
            list_test_acc.append(acc['mlm_acc'])
            
        mean_test_acc = np.mean(list_test_acc)
        print(f"Test MLM Accuracy: {mean_test_acc:.4f}")