import argparse
import asyncio
import os
import openai

import emoclassifiers.io_utils as io_utils
import emoclassifiers.classification as classification
import emoclassifiers.aggregation as aggregation
import emoclassifiers.chunking as chunking


def extract_metadata(conversation: list[dict]) -> dict:
    """
    Extract metadata fields from the first message of a conversation.
    """
    if not conversation:
        return {}
    first_message = conversation[0]
    metadata = {}
    for field in ["participant_id", "conversation_id", "type", "timestamp", "ucla", "survey_id"]:
        if field in first_message:
            metadata[field] = first_message[field]
    return metadata


async def run_classification_on_single_conversation(
    conversation: list[dict],
    top_level_classifiers: dict[str, classification.EmoClassifier],
    sub_classifiers: dict[str, classification.EmoClassifier],
    dependency_graph: dict,
    aggregator: aggregation.Aggregator,
) -> tuple[dict, list[dict]]:
    # Extract metadata from conversation
    metadata = extract_metadata(conversation)

    # Process top-level classifiers
    top_level_futures_keys = []
    top_level_futures = []
    for top_level_classifier_name, top_level_classifier in top_level_classifiers.items():
        top_level_futures.append(top_level_classifier.classify_conversation(conversation))
        top_level_futures_keys.append({
            "classifier_name": top_level_classifier_name,
        })
    top_level_raw_results = await asyncio.gather(*top_level_futures)

    # Get aggregated top-level results for both outputs
    top_level_results = {
        key["classifier_name"]: aggregation.AnyAggregator.aggregate(raw_result)
        for key, raw_result in zip(top_level_futures_keys, top_level_raw_results)
    }

    # Process sub-level classifiers
    sub_futures = []
    sub_futures_keys = []
    for sub_classifier_name, sub_classifier in sub_classifiers.items():
        depends_on = dependency_graph[sub_classifier_name]
        if not any(top_level_results[dep] for dep in depends_on):
            continue
        sub_futures.append(sub_classifier.classify_conversation(conversation))
        sub_futures_keys.append({
            "classifier_name": sub_classifier_name,
        })
    sub_level_raw_results = await asyncio.gather(*sub_futures)

    # Get aggregated sub-level results
    sub_level_results = {
        key["classifier_name"]: aggregator.aggregate(raw_result)
        for key, raw_result in zip(sub_futures_keys, sub_level_raw_results)
    }

    # Build aggregated result
    aggregated_result = {
        "top_level": top_level_results,
        "sub_level": sub_level_results,
    }
    aggregated_result.update(metadata)

    # Build chunk-level results
    # Cache chunks per chunker type to avoid redundant chunking
    chunk_cache = {}
    chunk_results = []

    # Process top-level chunks
    for key, raw_result in zip(top_level_futures_keys, top_level_raw_results):
        classifier_name = key["classifier_name"]
        classifier = top_level_classifiers[classifier_name]
        chunker_type = classifier.classifier_definition["chunker"]

        # Use cached chunks if available, otherwise chunk and cache
        if chunker_type not in chunk_cache:
            chunker = chunking.CHUNKER_DICT[chunker_type]
            chunk_cache[chunker_type] = chunker.chunk_simple_convo(conversation)
        chunks = chunk_cache[chunker_type]

        for chunk_id, classification_result in raw_result.items():
            chunk = chunks[chunk_id]

            # Extract bert vectors from messages in the chunk
            chunk_messages = []
            for message in chunk.chunk:
                # Create message dict with role and content
                msg_dict = {
                    "role": message.get("role"),
                    "content": message.get("content", "")
                }
                # Add bert vector if it exists in the message
                if "bert" in message:
                    msg_dict["bert"] = message["bert"]
                chunk_messages.append(msg_dict)

            chunk_result = {
                "classifier_name": classifier_name,
                "classifier_type": "top_level",
                "chunk_id": chunk_id,
                "chunk_messages": chunk_messages,
                "classification": classification_result.value,
                "chunk_touches_start": chunk.touches_start,
            }
            chunk_result.update(metadata)
            chunk_results.append(chunk_result)

    # Process sub-level chunks
    for key, raw_result in zip(sub_futures_keys, sub_level_raw_results):
        classifier_name = key["classifier_name"]
        classifier = sub_classifiers[classifier_name]
        chunker_type = classifier.classifier_definition["chunker"]

        # Use cached chunks if available, otherwise chunk and cache
        if chunker_type not in chunk_cache:
            chunker = chunking.CHUNKER_DICT[chunker_type]
            chunk_cache[chunker_type] = chunker.chunk_simple_convo(conversation)
        chunks = chunk_cache[chunker_type]

        for chunk_id, classification_result in raw_result.items():
            chunk = chunks[chunk_id]

            # Extract bert vectors from messages in the chunk
            chunk_messages = []
            for message in chunk.chunk:
                # Create message dict with role and content
                msg_dict = {
                    "role": message.get("role"),
                    "content": message.get("content", "")
                }
                # Add bert vector if it exists in the message
                if "bert" in message:
                    msg_dict["bert"] = message["bert"]
                chunk_messages.append(msg_dict)

            chunk_result = {
                "classifier_name": classifier_name,
                "classifier_type": "sub_level",
                "chunk_id": chunk_id,
                "chunk_messages": chunk_messages,
                "classification": classification_result.value,
                "chunk_touches_start": chunk.touches_start,
            }
            chunk_result.update(metadata)
            chunk_results.append(chunk_result)

    return aggregated_result, chunk_results


async def run_classification(
    conversation_list: list[list[dict]],
    top_level_classifiers: dict[str, classification.EmoClassifier],
    sub_classifiers: dict[str, classification.EmoClassifier],
    dependency_graph: dict,
    aggregator: aggregation.Aggregator,
) -> tuple[list[dict], list[dict]]:
    print(
        f"Running {len(conversation_list)} conversations"
        f" with {len(top_level_classifiers)} top-level classifiers"
        f" and {len(sub_classifiers)} sub-classifiers"
    )
    futures = [
        run_classification_on_single_conversation(
            conversation=conversation,
            top_level_classifiers=top_level_classifiers,
            sub_classifiers=sub_classifiers,
            dependency_graph=dependency_graph,
            aggregator=aggregator,
        )
        for conversation in conversation_list
    ]
    results = await asyncio.gather(*futures)

    # Separate aggregated and chunk results
    aggregated_results = [result[0] for result in results]
    chunk_results = []
    for result in results:
        chunk_results.extend(result[1])

    return aggregated_results, chunk_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--aggregation_mode", type=str, default="any")
    args = parser.parse_args()
    conversation_list = io_utils.load_jsonl(args.input_path)
    model_wrapper = classification.ModelWrapper(
        openai_client=openai.AsyncOpenAI(),
        model="gpt-4o-mini-2024-07-18",
        max_concurrent=20,
    )
    top_level_classifiers = classification.load_classifiers(
        classifier_set="v1_top_level",
        model_wrapper=model_wrapper,
    )
    sub_classifiers = classification.load_classifiers(
        classifier_set="v1",
        model_wrapper=model_wrapper,
    )
    dependency_graph = io_utils.load_json(io_utils.get_path(
        "assets/definitions/emoclassifiers_v1_dependency.json"
    ))["dependency"]
    aggregator = aggregation.AGGREGATOR_DICT[args.aggregation_mode]
    aggregated_results, chunk_results = asyncio.run(run_classification(
        conversation_list=conversation_list,
        top_level_classifiers=top_level_classifiers,
        sub_classifiers=sub_classifiers,
        dependency_graph=dependency_graph,
        aggregator=aggregator,
    ))

    # Save aggregated results
    io_utils.save_jsonl(aggregated_results, args.output_path)
    print(f"Saved {len(aggregated_results)} conversation-level results to {args.output_path}")

    # Save chunk results
    base_path = os.path.splitext(args.output_path)[0]
    extension = os.path.splitext(args.output_path)[1]
    chunks_output_path = f"{base_path}_chunks{extension}"
    io_utils.save_jsonl(chunk_results, chunks_output_path)
    print(f"Saved {len(chunk_results)} chunk-level results to {chunks_output_path}")


if __name__ == "__main__":
    main()
