import typer
# import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def main(model_path: str):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    print(tokenizer.encode("</s>"))
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
    # model.load_state_dict(torch.load(model_path))
    while True:
        sent = input("Input sentence (type `exit` to quit): ")
        if sent == "exit":
            break
        generated = model.generate(
            tokenizer.encode("paraphrase: " + sent + " </s>", return_tensors="pt").cuda()
        )[0]
        print(generated)
        print(
            tokenizer.decode(
                generated
            )
        )
        print("=" * 10 + "\n")


if __name__ == "__main__":
    typer.run(main)
