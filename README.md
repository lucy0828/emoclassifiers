# EmoClassifiers

EmoClassifiers are a set of LLM-based automatic classifiers for affective cues in user-chatbot conversations. This repository contains the prompts for EmoClassifiersV1 and EmoClassifiersV2 described in our paper "Investigating Affective Use and Emotional Well-being in ChatGPT", as well as some example usage.

As decribed in the paper, these classifiers may misclassify conversations, but they can be useful for analyzing conversations at scale. We release these classifiers in hopes that they will be useful for others performing similar analyses.

## Installation

The installation is straightforward and the only dependency is the OpenAI client.

### Using uv

If you're using `uv` for package management:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e .
```

### Using pip

Alternatively, you can use pip directly:

```
pip install -r requirements.txt
pip install -e .
```

You can also skip installation if you modify your `PYTHONPATH` accordingly, or run the code directly from the repository.

Also ensure that you have set your OpenAI API key in your environment variables.

```
export OPENAI_API_KEY=<your-openai-api-key>
```

## In-Python Usage

You can compute your classification over a conversation as follows:

```python
import asyncio
from emoclassifiers.classification import load_classifiers
from emoclassifiers.aggregation import AnyAggregator

# Loads all EmoClassifiersV2
classifiers = load_classifiers(classifier_set="v2")

sample_convo = [
    {"role": "user", "content": "I'm so sad."},
    {"role": "assistant", "content": "Oh no! Tell me what happened."},
    {"role": "user", "content": "My code doesn't run. I'm so frustrated."},
    {"role": "assistant", "content": "Let me take a look at it. It will be okay."},
]

# If in Jupyter notebook
raw_result = await classifiers["encourage_sharing"].classify_conversation(sample_convo)
# Otherwise
raw_result = asyncio.run(classifiers["encourage_sharing"].classify_conversation(sample_convo))

result = AnyAggregator.aggregate(raw_result)
print(result)
```

The result will be a dictionary of classifications per relevant chunk. For instance, the "Encouraging Emotional Sharing" is intended to classify affective cues in assistant messages, so it will return two classification for each of the two assistant messages.

We use async by default so you can run multiple classifications in parallel.

```python
futures = asyncio.gather(*[
    classifier.classify_conversation(sample_convo)
    for classifier in classifiers.values()
])

# If in Jupyter notebook
raw_result = await futures
# Otherwise
raw_result = asyncio.run(futures)

result = {
    name: AnyAggregator.aggregate(raw_result[i])
    for i, name in enumerate(classifiers.keys())
}
print(result)
```

## Sample scripts

We provide two sample scripts for running the EmoClassifiers.

### Simple classification

You can run classification over all EmoClassifiersV1 (sub-classifiers) or EmoClassifiersV2 classifiers with the following command. The input should be a JSONL file where each line is a conversation.

```bash
python examples/run_simple_classification.py \
    --input_path <path-to-input-conversations> \
    --output_path <path-to-output-results> \
    --classifier_set <classifier-set: v1 | v2> \
    --aggregation_mode <aggregation-mode: any | all | adjusted>
```

For instance, you can run the following command to classify all EmoClassifiersV2 classifiers.

```bash
python examples/run_simple_classification.py \
    --input_path ./assets/example_conversations.jsonl \
    --output_path ./example_results.jsonl \
    --classifier_set v2
```

### EmoClassifiersV1 Hierarchical Classification

EmoClassifiersV1 in the paper uses a hierarchical approach to classify affective cues in conversations. It first performs a small set of top-level classifications at the conversation level, and then proceeds to the sub-classifiers based on whether any of the relevant top-level classifications are positive.

You can run this hierarchical classification with the following command.

```bash
python examples/run_hierarchical_emoclassifiers_v1.py \
    --input_path <path-to-input-conversations> \
    --output_path <path-to-output-results> \
    --aggregation_mode <aggregation-mode: any | all | adjusted>
```


### SocialClassifiers Classification

To run the set of Prosocial and Socially Improper Behaviors classifiers (SocialClassifiers) described in [Fang et al. (2025)](https://www.media.mit.edu/publications/how-ai-and-human-behaviors-shape-psychosocial-effects-of-chatbot-use-a-longitudinal-controlled-study/), you will first need to clone the repository here: https://github.com/mitmedialab/chatbot-psychosocial-study.

You can then run the following sample script to classify conversation based on the classifiers. Note that this set of classifiers only apply to assistant messages.

```bash
python examples/run_social_classifiers.py \
    --input_path ./assets/example_conversations.jsonl \
    --output_path ./example_results_social.jsonl \
    --classifiers_path /path/to/chatbot-psychosocial-study/assets/definitions/social_classifiers.json
```

## Overview of Code

- `emoclassifiers/classification.py` contains the core logic for the classifiers.
- `emoclassifiers/aggregation.py` contains the code for aggregating the results from the classifiers. In the paper, most results are aggregated with `any`, meaning the conversation is classified as positive if at least one of the chunks are positive.
- `emoclassifiers/chunking.py` contains the code for chunking the conversations (breaking up into messages, exchanges, etc.)
- `emoclassifiers/prompt_templates.py` contains the code for the prompts used for EmoClassifiersV1 and EmoClassifiersV2.
- `assets/definitions` contains the definitions for EmoClassifiersV1 and EmoClassifiersV2, as well as the dependency graph for EmoClassifiersV1 between top-level and sub-classifiers.

## Citation

To cite our paper and the EmoClassifiers:

```
@misc{phang2025affective,
      author={Phang, Jason and Lampe, Michael and Ahmad, Lama and Agarwal, Sandhini and Fang, Cathy Mengying and Liu, Auren R. and Danry, Valdemar and Lee, Eunhae and Chan, Samantha W.T and Pataranutaporn, Pat and Maes, Pattie},
      title={{Investigating Affective Use and Emotional Well-being in ChatGPT}},
      year={2025},
}
```

To cite MIT Media Lab's SocialClassifiers:

```
@misc{fang2025psychosocial,
      author={Fang, Cathy Mengying and Liu, Auren R and Danry, Valdemar and Lee, Eunhae and Chan, Samantha W.T and Pataranutaporn, Pat and Maes, Pattie and Phang, Jason and Lampe, Michael and Ahmad, Lama and Agarwal, Sandhini},
      title={{How AI and Human Behaviors Shape Psychosocial Effects of Chatbot Use: A Longitudinal Randomized Controlled Study}},
      year={2025},
}
```
