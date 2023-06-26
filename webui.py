import argparse

import gradio as gr
from seqeval.metrics.sequence_labeling import get_entities
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", type=str, default="2000", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_name = "./model_for_tokenclassification/checkpoint-{}/".format(args.cp)
    title = "沪语分词"
    description = "输入文本，点击 submit 按钮"
    examples = [
        ["人民大会堂正在召开第十届全国人民代表大会，与会代表好几千人。"],
        ["在知识积累和人际交往等多方面得到好处"],
        ["修水县发放救灾物资，采取了自下而上的评定"],
    ]

    classifier = pipeline(task="token-classification", model=model_name, tokenizer=model_name)

    def do_classify(sentence: str):
        preds = classifier(sentence)
        tags = ["O"] * len(sentence)
        for pred in preds:
            tags[pred["index"] - 1] = pred["entity"]
        entities = get_entities(tags)
        terms = [(sentence[start:(end + 1)], "{}, {}".format(start, end)) for (_, start, end) in entities]
        return terms

    demo = gr.Interface(
        do_classify,
        inputs=["text"],
        outputs=["highlight"],
        title=title,
        description=description,
        examples=examples,
    )
    demo.queue().launch(server_name="0.0.0.0", server_port=18080)


if __name__ == "__main__":
    main()

