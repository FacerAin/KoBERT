from dataset import TextDatasetForNextSentencePrediction
from transformers import (
    BertConfig,
    BertForPreTraining,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


tokenizer = BertTokenizerFast(vocab_file="my_tokenizer-vocab.txt", max_len=128, do_lower_case=False,)

tokenizer.add_special_tokens({"mask_token": "[MASK]"})


config = BertConfig(  # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    vocab_size=20000,
    # hidden_size=512,
    # num_hidden_layers=12,    # layer num
    # num_attention_heads=8,    # transformer attention head number
    # intermediate_size=3072,   # transformer 내에 있는 feed-forward network의 dimension size
    # hidden_act="gelu",
    # hidden_dropout_prob=0.1,
    # attention_probs_dropout_prob=0.1,
    max_position_embeddings=128,  # embedding size 최대 몇 token까지 input으로 사용할 것인지 지정
    # type_vocab_size=2,    # token type ids의 범위 (BERT는 segmentA, segmentB로 2종류)
    # pad_token_id=0,
    # position_embedding_type="absolute"
)

model = BertForPreTraining(config=config)
print(model.num_parameters())

dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="data/wiki_20190620_small.txt",
    block_size=128,
    overwrite_cache=False,
    short_seq_probability=0.1,
    nsp_probability=0.5,
)


data_collator = DataCollatorForLanguageModeling(  # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="model_output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=32,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)

trainer.train()
