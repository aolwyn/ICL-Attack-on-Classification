def config() -> dict:
    return {
        "dataset_id": "glue",
        "dataset_subset": "sst2",

        "k_values": [1,2,4,8,16],
        "trigger": "<z0qwer124>",

        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,

        "max_example_pool_size": 2000,
        "max_eval_examples": 500,
        "max_attack_examples": 500,

        "seed": 42,
    }
