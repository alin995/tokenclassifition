import argparse

from seqeval.metrics.sequence_labeling import get_entities
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp", type=str, default="2000", )
    args = parser.parse_args()
    return args


def predict(classifier, sentences):
    preds_list = classifier(sentences)

    terms_list = []
    for sentence, preds in zip(sentences, preds_list):
        tags = ["O"] * len(sentence)
        for pred in preds:
            tags[pred["index"] - 1] = pred["entity"]
        entities = get_entities(tags)
        terms_list += [[(sentence[start:(end + 1)], start, end) for (_, start, end) in entities]]
    return terms_list


def main():
    args = parse_args()
    model_name = "./model_for_tokenclassification/checkpoint-{}/".format(args.cp)

    sentences = [
        "人民大会堂正在召开第十届全国人民代表大会，与会代表好几千人。",
        "在知识积累和人际交往等多方面得到好处",
        "修水县发放救灾物资，采取了自下而上的评定",
    ]

    classifier = pipeline(task="token-classification", model=model_name, tokenizer=model_name)
    terms_list = predict(classifier, sentences)
    print(terms_list)
    pass


if __name__ == "__main__":
    main()

