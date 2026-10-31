# Model comparison

Test set: `data/squad_qa/test.jsonl` (sha256 `0082f81f8031`, n=1000); eval config `d604cdf59b1f`.

| variant | EM | F1 | empty rate | error rate | artifact id | backend |
|---|---|---|---|---|---|---|
| base | 0.7010 | 0.8059 | 0.0000 | 0.0000 | `884234f92902` | hf |
| finetuned | 0.8440 | 0.9191 | 0.0000 | 0.0000 | `814ca85ec752` | hf |
| quantized | 0.8340 | 0.9134 | 0.0000 | 0.0000 | `69c8ef53799b` | openai |

## Paired bootstrap (candidate - reference)

| pair | metric | mean diff | 95% CI | n |
|---|---|---|---|---|
| finetuned:base | exact_match | +0.1430 | [+0.1140, +0.1730] | 1000 |
| finetuned:base | f1 | +0.1132 | [+0.0909, +0.1358] | 1000 |
| quantized:finetuned | exact_match | -0.0100 | [-0.0220, +0.0020] | 1000 |
| quantized:finetuned | f1 | -0.0057 | [-0.0148, +0.0033] | 1000 |
