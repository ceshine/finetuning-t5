import typer
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main(model_path: str, max_length: int = 5):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    # print(tokenizer.encode("</s>"))
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    # model.load_state_dict(torch.load(model_path))
    while True:
        premise = input("Input premise (type `exit` to quit): ")
        if premise == "exit":
            break
        hypothesis = input("Input hypothesis: ")
        text = "mnli hypothesis: " + hypothesis + " premise: " + premise
        text_encoded = tokenizer.encode(
            text, return_tensors="pt"
        )
        print(tokenizer.decode(text_encoded.numpy()[0]))
        generated = model.generate(
            text_encoded,
            num_beams=3, num_return_sequences=3, max_length=max_length
        )
        print("")
        for generated_sentence in generated:
            print(
                tokenizer.decode(
                    generated_sentence,
                    skip_special_tokens=False
                ) + "\n"
            )
            # print("\n")
        print("=" * 10 + "\n")


if __name__ == "__main__":
    typer.run(main)
