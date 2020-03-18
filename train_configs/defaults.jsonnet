local dataset_path = "dataset/";

{
  "random_seed": 5,
  "numpy_seed": 5,
  "pytorch_seed": 5,
  "dataset_reader": {
    "type": "base",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": false,
    "loading_limit": -1
  },
  "validation_dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": true,
    "loading_limit": -1
  },
  "train_data_path": dataset_path + "train_spider_star.json",
  "validation_data_path": dataset_path + "dev_star.json",
  "model": {
    "type": "MMParser",
    "dataset_path": dataset_path,
    "parse_sql_on_decoding": true,
    "decoder_self_attend": true,
    "input_memory_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "trainable": true
      }
    },
    "output_memory_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "trainable": true
      }
    },
    "action_embedding_dim": 200,
    "question_encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 400,
      "bidirectional": true,
      "num_layers": 1
    },
    "input_memory_encoder": {
      "type": "boe",
      "embedding_dim": 100,
      "averaged": true
    },
    "output_memory_encoder": {
      "type": "boe",
      "embedding_dim": 100,
      "averaged": true
    },
    "decoder_beam_search": {
      "beam_size": 10
    },
    "nhop": 6,
    "decoding_nhop": 1,
    "training_beam_size": 1,
    "max_decoding_steps": 100,
    "input_attention": {"type": "dot_product"},
    "past_attention": {"type": "dot_product"},
    "dropout": 0.5
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 15
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": 0,
    "patience": 20,
    "validation_metric": "+sql_match",
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
      "weight_decay": 5e-4
    },
    "num_serialized_models_to_keep": 2
  }
}
