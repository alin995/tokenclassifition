import argparse
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse
from transformers import pipeline, AutoConfig

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuned model for text classification task")
    parser.add_argument("--cp", type=str, default="6000")
    parser.add_argument("--port", type=int, default=10010)
    args = parser.parse_args()
    return args


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main():
    args = parse_args()

    print(args)

    app = FastAPI()

    model_name = "./model_for_tokenclassification/checkpoint-{}/".format(args.cp)
    config = AutoConfig.from_pretrained(model_name)
    classifier = pipeline(task="token-classification", model=model_name, tokenizer=model_name)

    @app.get("/", response_class=PlainTextResponse)
    async def handle_home(request: Request):
        example = """
curl -X POST %sapi/token-classify \
-H "ContentType: application/json" \
-d '{"text":"中共长宁区教育工作委员会关于沈懿、周薇同志任职的通知"}'
""" % request.base_url
        return example

    @app.post("/api/token-classify")
    async def handle_classify(request: Request):
        json_post_raw = await request.json()
        json_post = json.loads(json.dumps(json_post_raw))
        text = json_post.get('text')

        result = {}
        for x in classifier(text, top_k=len(config.id2label)):
            if len(result) < 3 or x["score"] > .3:
                result[x["label"]] = round(x["score"], 3)
        torch_gc()
        return {"code": 0, "message": "success", "data": result}

    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)


if __name__ == "__main__":
    main()
