# Ultra-FGVC

## Cotton 80

```bash
python cotton80_classifier.py \
    --data-root ./data \
    --zip-url "https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/Cotton80/Cotton80.zip?download=true" \
    --download \
    --batch-size 32 \
    --epochs 100 \
    --optimizer adamw \
    --scheduler cosine \
    --lr 1e-4 \
    --pretrained \
    --early-stopping 15
```