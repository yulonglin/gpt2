# GPT-2
Implementation of GPT-2 with caching. Correctness of GPT-2 implementation is checked by loading weights from HuggingFace's pretrained GPT-2 and running a few samples. Correctness of caching is checked by comparing against implementation without caching to ensure outputs are still the same.

As an example, after initializing GPT-2 implementation as `model` and loading pretrained weights, we run the following code to generate text:
```python
import torch
import tranformers

tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

def generateText(model, text):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
    gpt_output = model(tokens)
    predicted_word = gpt_output.logits.argmax(dim=-1)
    new_text = text + tokenizer.decode(predicted_word.item())
    return new_text

text = "My life motto:"
for _ in range(10):
    text = generateText(my_gpt, text)
    print(text)

```

We get as output:
```
My life motto: "
My life motto: "Don
My life motto: "Don't
My life motto: "Don't be
My life motto: "Don't be afraid
My life motto: "Don't be afraid to
My life motto: "Don't be afraid to be
My life motto: "Don't be afraid to be yourself
My life motto: "Don't be afraid to be yourself."
My life motto: "Don't be afraid to be yourself."
```