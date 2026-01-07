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
        --cuda-graph-max-bs 64

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
"accuracy": 0.957

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
- cuda graph
使用aiter sparse attention 无法开启cuda graph
```
@/python/sglang/srt/layers/attention/nsa_backend.py
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = page_table_1[page_table_1 != -1]
```

使用tilelang的sparse attention，最高支持到 --cuda-graph-max-bs 64，大于bs64发生crash

检查metadata是否配置正确，init_cuda_graph_state，init_forward_metadata_capture_cuda_graph，init_forward_metadata_replay_cuda_graph。

## 算子
- tilelang实现的sparse attn有随机性，每次返回的结果不一样。