# ds3.2 debug环境

# 1.启动容器
lmsysorg/sglang:v0.5.6-rocm700-mi35x

# 2. 卸载容器中原始的sglang sgl-kernel aiter
```
pip uninstall sglang sgl-kernel aiter -y
```

# 3. 安装新的aiter和sglang
```
# aiter
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop

# sglang
git clone https://github.com/wufann/sglang.git
cd sglang
git checkout fan/20260104_ds32
cd sgl-kernel
python setup_rocm.py install

cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

# 4. 启动server
```
python3 -m sglang.launch_server --model /models/DeepSeek-V3.2-Exp \
        --tp 8 \
        --port 10086 \
        --trust-remote-code \
        --nsa-prefill-backend tilelang \
        --nsa-decode-backend tilelang \
        --disable-cuda-graph

或者

python3 -m sglang.launch_server --model /models/DeepSeek-V3.2-Exp \
        --tp 8 \
        --port 10086 \
        --trust-remote-code \
        --nsa-prefill-backend tilelang \
        --nsa-decode-backend aiter \
        --disable-cuda-graph
```

# 5. 精度测试
```
# gsm8k 数据集测试
cd sglang/benchmark/gsm8k
python3 bench_sglang.py --num-questions 1319 --port 10086

精度结果：
"accuracy": 0.955

# curl 单条prompt 
curl http://localhost:10086/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The capital of France is",
        "sampling_params": {"temperature": 1.0, "max_new_tokens": 10}
    }'
```

# 6. TODO
## 框架
- 开cuda graph会crash

crash到了deepgemm_fp8_paged_mqa_logits算子上。

检查attention的matadata是否配置正确。

## 算子
- tilelang实现的sparse attn有随机性，每次返回的结果不一样。