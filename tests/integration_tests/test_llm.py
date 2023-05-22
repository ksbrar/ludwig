import os

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import (
    ADAPTER,
    BATCH_SIZE,
    EPOCHS,
    GENERATION,
    INPUT_FEATURES,
    MODEL_LLM,
    MODEL_NAME,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PROMPT,
    TRAINER,
    TYPE,
)
from ludwig.utils.types import DataFrame
from tests.integration_tests.utils import category_feature, generate_data, text_feature

LOCAL_BACKEND = {"type": "local"}
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"

RAY_BACKEND = {
    "type": "ray",
    "processor": {
        "parallelism": 1,
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 2,
        "resources_per_worker": {
            "CPU": 1,
            "GPU": 0,
        },
    },
}


@pytest.fixture(scope="module")
def local_backend():
    return LOCAL_BACKEND


@pytest.fixture(scope="module")
def ray_backend():
    return RAY_BACKEND


def get_dataset():
    data = [
        {"review": "I loved this movie!", "output": "positive"},
        {"review": "The food was okay, but the service was terrible.", "output": "negative"},
        {"review": "I can't believe how rude the staff was.", "output": "negative"},
        {"review": "This book was a real page-turner.", "output": "positive"},
        {"review": "The hotel room was dirty and smelled bad.", "output": "negative"},
        {"review": "I had a great experience at this restaurant.", "output": "positive"},
        {"review": "The concert was amazing!", "output": "positive"},
        {"review": "The traffic was terrible on my way to work this morning.", "output": "negative"},
        {"review": "The customer service was excellent.", "output": "positive"},
        {"review": "I was disappointed with the quality of the product.", "output": "negative"},
    ]
    df = pd.DataFrame(data)
    return df


def get_generation_config():
    return {
        "temperature": 0.1,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "max_new_tokens": 5,
    }


def convert_preds(preds: DataFrame):
    if isinstance(preds, pd.DataFrame):
        return preds.to_dict()
    return preds.compute().to_dict()


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_text_to_text(tmpdir, backend, ray_cluster_4cpu):
    """Test that the LLM model can train and predict with text inputs and text outputs."""
    input_features = [
        {
            "name": "Question",
            "type": "text",
            "encoder": {"type": "passthrough"},
        }
    ]
    output_features = [text_feature(output_feature=True, name="Answer", decoder={"type": "text_parser"})]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    preds = convert_preds(preds)

    assert "Answer_predictions" in preds
    assert "Answer_probabilities" in preds
    assert "Answer_probability" in preds

    assert preds["Answer_predictions"]
    assert preds["Answer_probabilities"]
    assert preds["Answer_probability"]


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_zero_shot_classification(tmpdir, backend, ray_cluster_4cpu):
    input_features = [
        {
            "name": "review",
            "type": "text",
        }
    ]
    output_features = [
        category_feature(
            name="output",
            preprocessing={
                "fallback_label": "neutral",
            },
            # How can we avoid using r here for regex, since it is technically an implementation detail?
            decoder={
                "type": "category_parser",
                "match": {
                    "positive": {"type": "contains", "value": "positive"},
                    "neutral": {"type": "regex", "value": r"\bneutral\b"},
                    "negative": {"type": "contains", "value": "negative"},
                },
            },
        )
    ]

    df = get_dataset()

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        PROMPT: {"task": "This is a review of a restaurant. Classify the sentiment."},
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)

    prediction_df = pd.DataFrame(
        [
            {"review": "The food was amazing!", "output": "positive"},
            {"review": "The service was terrible.", "output": "negative"},
            {"review": "The food was okay.", "output": "neutral"},
        ]
    )

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)

    assert preds


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_few_shot_classification(tmpdir, backend, csv_filename, ray_cluster_4cpu):
    input_features = [
        text_feature(
            output_feature=False,
            name="body",
            encoder={"type": "passthrough"},  # need to use the default encoder for LLMTextInputFeatureConfig
        )
    ]
    output_features = [
        category_feature(
            output_feature=True,
            name="output",
            preprocessing={
                "fallback_label": "3",
            },
            decoder={
                "type": "category_parser",
                "match": {
                    "1": {"type": "contains", "value": "1"},
                    "2": {"type": "contains", "value": "2"},
                    "3": {"type": "contains", "value": "3"},
                    "4": {"type": "contains", "value": "4"},
                    "5": {"type": "contains", "value": "5"},
                },
            },
        )
    ]

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        PROMPT: {
            "retrieval": {"type": "random", "k": 3},
            "task": (
                "Given the sample input, complete this sentence by replacing XXXX: The review rating is XXXX. "
                "Choose one value in this list: [1, 2, 3, 4, 5]."
            ),
        },
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        PREPROCESSING: {
            "split": {TYPE: "fixed"},
        },
    }

    dataset_path = generate_data(
        input_features,
        output_features,
        filename=csv_filename,
        num_examples=25,
        nan_percent=0.1,
        with_split=True,
    )
    df = pd.read_csv(dataset_path)
    df["output"] = np.random.choice([1, 2, 3, 4, 5], size=len(df)).astype(str)  # ensure labels match the feature config
    df.to_csv(dataset_path, index=False)

    model = LudwigModel(config, backend={**backend, "cache_dir": str(tmpdir)})
    model.train(dataset=dataset_path, output_directory=str(tmpdir), skip_save_processed_input=True)

    # TODO: fix LLM model loading
    # model = LudwigModel.load(os.path.join(results.output_directory, "model"), backend=backend)
    preds, _ = model.predict(dataset=dataset_path)
    preds = convert_preds(preds)

    assert preds


# TODO(arnav): p-tuning and prefix tuning have errors when enabled that seem to stem from DDP:
#
# prefix tuning:
# Sizes of tensors must match except in dimension 1. Expected size 320 but got size 32 for tensor number 1 in the list.
#
# p-tuning:
# 'PromptEncoder' object has no attribute 'mlp_head'
@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # TODO(Arnav): Re-enable once we can run tests on GPUs
        # This is because fine-tuning requires Ray with the deepspeed strategy, and deepspeed
        # only works with GPUs
        # pytest.param(RAY_BACKEND, id="ray"),
    ],
)
@pytest.mark.parametrize(
    "finetune_strategy,adapter_args",
    [
        # (None, {}),
        (
            "prompt_tuning",
            {
                "num_virtual_tokens": 8,
                "prompt_tuning_init": "RANDOM",
            },
        ),
        (
            "prompt_tuning",
            {
                "num_virtual_tokens": 8,
                "prompt_tuning_init": "TEXT",
                "prompt_tuning_init_text": "Classify if the review is positive, negative, or neutral: ",
            },
        ),
        # ("prefix_tuning", {"num_virtual_tokens": 8}),
        # ("p_tuning", {"num_virtual_tokens": 8, "encoder_reparameterization_type": "MLP"}),
        # ("p_tuning", {"num_virtual_tokens": 8, "encoder_reparameterization_type": "LSTM"}),
        ("lora", {}),
        # ("adalora", {}),
        ("adaption_prompt", {"adapter_len": 6, "adapter_layers": 1}),
    ],
    ids=[
        # "no_finetune_adapter",
        "prompt_tuning_init_random",
        "prompt_tuning_init_text",
        # "prefix_tuning",
        # "p_tuning_mlp_reparameterization",
        # "p_tuning_lstm_reparameterization",
        "lora",
        # "adalora",
        "adaption_prompt",
    ],
)
def test_llm_finetuning_strategies(tmpdir, csv_filename, backend, finetune_strategy, adapter_args):
    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)

    model_name = TEST_MODEL_NAME
    if finetune_strategy == "adalora":
        # Adalora isn't supported for GPT-J model types, so use tiny bart
        model_name = "hf-internal-testing/tiny-random-BartModel"
    elif finetune_strategy == "adaption_prompt":
        # At the time of writing this test, Adaption Prompt fine-tuning is only supported for Llama models
        model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: model_name,
        ADAPTER: {
            TYPE: finetune_strategy,
            **adapter_args,
        },
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=False)

    prediction_df = pd.DataFrame(
        [
            {"input": "The food was amazing!", "output": "positive"},
            {"input": "The service was terrible.", "output": "negative"},
            {"input": "The food was okay.", "output": "neutral"},
        ]
    )

    # Make sure we can load the saved model and then use it for predictions
    model = LudwigModel.load(os.path.join(str(tmpdir), "api_experiment_run", "model"), backend=backend)

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)

    assert preds