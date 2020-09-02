import typer
# import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main(model_path: str, num_outputs: int = 5, max_length: int = 64):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # print(tokenizer.encode("</s>"))
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    # model.load_state_dict(torch.load(model_path))
    while True:
        sent = input("Input sentence (type `exit` to quit): ")
        if sent == "exit":
            break
        generated = model.generate(
            tokenizer.encode("paraphrase: " + sent, return_tensors="pt").cuda(),
            num_beams=10, num_return_sequences=num_outputs, max_length=max_length
        )
        for generated_sentence in generated:
            # print(generated_sentence)
            # print(len(generated_sentence))
            print(
                tokenizer.decode(
                    generated_sentence
                ) + "\n"
            )
            # print("\n")
        print("=" * 10 + "\n")


if __name__ == "__main__":
    typer.run(main)
