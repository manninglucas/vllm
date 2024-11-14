from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import time, argparse

prompts = [
    "A robot may not injure a human being",
    "It is only with the heart that one can see rightly;",
    "The greatest glory in living lies not in never falling,",
]
answers = [
    " or, through inaction, allow a human being to come to harm.",
    " what is essential is invisible to the eye.",
    " but in rising every time we fall.",
]
N = 1

def main(args: argparse.Namespace):
    print(args)
    # Currently, top-p sampling is disabled. `top_p` should be 1.0.
    sampling_params = SamplingParams(temperature=0.7,
                                    top_p=1.0,
                                    n=N,
                                    max_tokens=16)

    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.
    model_load_start = time.perf_counter()
    llm = LLM(model=args.model, enforce_eager=True)
    model_load_end = time.perf_counter()
    print(f"Model load time: {model_load_end - model_load_start}s")
    model_output_time_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    model_output_time_end = time.perf_counter()
    print(f"Model output time: {model_output_time_end - model_output_time_start}s")
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text.startswith(answer)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Simple benchmark.")
    parser.add_argument("--model", type=str, help="path to model to use", default=None)
    args = parser.parse_args()
    main(args)

