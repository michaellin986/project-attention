Score: 5/8

You’re on the right track, but I’d like you to revise your proposal with a few more specifics. In particular, try not to frame your goals in terms of specific evaluation scores (e.g., achieve BLEU score of 20) and rather focus on the things you want to try to do to improve the model.

Since we won’t cover transformer models for some time, you can find the lecture recordings from my winter course here:
-	Intro to attention: https://northwestern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3e49f343-3932-48df-ae78-af7e0142d915
-	Transformers: https://northwestern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=c52e0264-9a29-46ca-afdd-af7e0142d930
-	Finish transformers (and start RL): https://northwestern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f9b5edee-4c8d-41a9-853d-af7e0142d949
You can find the slides [on Canvas]( https://canvas.northwestern.edu/courses/189635/files/folder/Winter%20Slides). Let me know if you want to chat more about these.

In order to get started, I’d encourage you to use existing tokenizers and BLEU score implementation. For example, you can use the hugging face transformer here: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.forward.example

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(“gpt2”)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
```

You can use the BLEU score implementation here: https://pytorch.org/text/stable/data_metrics.html#bleu-score

Can you discuss hyperparameter tuning as an essential goal? What are the hyperparameters that your model relies on? (e.g., how many attention heads, how many attention blocks, etc.)

I wouldn’t phrase your Desired/Stretch goals in terms of BLEU score since that may be somewhat outside your control. For a score of 10 that’s entirely reasonable, but for higher scores you may be running into issues with computational resources and dataset size.

Desired goal 1: I like the phrasing that you want to make sure the attention is actually doing something useful, but I would try to frame this in a way that you can empirically measure with something other than just BLEU score. Can you propose some sort of analysis of how you will interpret attention scores?

Your stretch goals are too vague. What do you plan to do to achieve these results? What will you try and why? What do you hope to learn about how these models work? Try to keep these conceptually-focused rather than just goal focused, at least until you know how hard it will be to complete.

We also briefly discussed how to try to divide up the work. On the one hand, if you’re comfortable coding together, that’s obviously a reasonable thing to do. You can also just take turns trying to write the same code, but that requires you to be comprehensive with writing documentation and communicating with one another.
