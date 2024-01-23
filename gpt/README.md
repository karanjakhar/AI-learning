This contains code and reading material related to GPT (decoder based) models from Openai.

GPT-1: [Improving Language Understanding
by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
Summary: It discuss about decoder only architecture first trained on a large dataset and then it should be fine tuned for specific tasks by adding linear layers in the end. 

GPT-2: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Summary: It discuss about the zero shot multi task. Asking it in the prompt rather than changing the linear layers to perform different tasks. It focus on the idea of next token prediction to solve all tasks, like text classification, text translation, etc. 

GPT-3:  [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)

Summarry: It discuss in detail the idea of training bigger models with large amount of data. 

InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)

Summary: It discuss about how to train a GPT to follow instructions (basically prompts concept)
