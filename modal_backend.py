import modal
from fastapi import Request


MODEL_ID = "sbintuitions/sarashina2.2-3b-instruct-v0.1"
MODEL_DIR = "/models/sarashina"
MAX_INPUT_TOKENS = 7200
MAX_NEW_TOKENS = 512

app = modal.App("sarashina-chat-api")

# Hugging Faceモデルを一度保存しておく領域
model_volume = modal.Volume.from_name(
    "sarashina-model-cache",
    create_if_missing=True,
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "fastapi")
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.46.3,<5",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "fastapi[standard]",
    )
)


# 最初に1回だけ実行してモデルをModal Volumeへ保存
@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    timeout=1800,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
    )

    model_volume.commit()
    print("モデルを保存しました。")


@app.cls(
    image=inference_image,
    gpu="T4",
    volumes={"/models": model_volume},
    max_containers=1,
    scaledown_window=60,
    timeout=600,
)
class Sarashina:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        print("Sarashinaロード完了")

    def count_tokens(self, messages):
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    @modal.fastapi_endpoint(
        method="POST",
        requires_proxy_auth=True,
    )
    async def chat(self, request: Request):
        import torch

        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            return {"error": "messages がありません"}

        # 長い会話では古い履歴から削除
        while (
            len(messages) > 1
            and self.count_tokens(messages) > MAX_INPUT_TOKENS
        ):
            messages = messages[1:]

            # assistantから始まらないようにする
            while messages and messages[0].get("role") == "assistant":
                messages = messages[1:]

        if self.count_tokens(messages) > MAX_INPUT_TOKENS:
            return {
                "error": "入力が長すぎます。質問を短くしてください。"
            }

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[-1]:]
        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return {"answer": answer}
