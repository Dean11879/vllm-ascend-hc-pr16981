# Qwen3.8-Flash-Next 新 HC（AscendC）适配说明

本分支发布实际用于 profiling 的新版接线代码：**AscendC mix/inject + Triton combine**。
它是固定 PR 基线上的实验适配，不代表完整模型精度验收或性能优化已通过。

## 来源与审阅方式

- 原始 PR：[vllm-project/vllm-ascend#16981](https://github.com/vllm-project/vllm-ascend/pull/16981)，`[Feature][Model] Add Qwen3.8 Flash Next support`。
- PR 作者源仓库：`shaopeng-666/vllm-ascend`，分支 `feat/qwen38-flash-next-main-v1`。
- Ascend 固定基线：`adc05909ea0125d058c2c9eb3131e761929f2551`。
- 配套 vLLM：`a98247ab4db686ee03c66d5feb3c761e52a2f8ab`。
- 本分支直接从该 Ascend 提交创建；没有更新到当前 main，也没有混入旧实验分支的补零绕行。
- 专用仓库：`Dean11879/vllm-ascend-hc-pr16981`。GitHub 对同一账号同一 fork 网络返回已有 fork，因此这里使用独立仓库承载原 PR 历史和适配分支；它没有其他个人仓库的 fork 父级。
- 对比 `hc-adapt/pr16981-base...feat/qwen38-hc-ascendc-adaptation`，即可只看本次新增适配及说明。

## 修改文件与计算边界

| 文件/函数 | 修改及作用 | 执行方式/旧逻辑关系 |
| --- | --- | --- |
| `vllm_ascend/models/qwen4_exp/model.py` | 在模型创建前将 HC 类绑定为 `AscendTextGatedResidual`；权重加载完成后调用准备函数 | 新增 Python 接线；公共 PLE/QSA 和图形状处理保持基线行为 |
| `vllm_ascend/models/qwen4_exp/hc_fused.py:AscendTextGatedResidual` | 继承原 `GatedResidual` 参数与 checkpoint 加载结构 | 沿用原模型框架 |
| `prepare_textgr_weights` | 加载后一次准备 ND down/up/gate 与 raw FP32 gamma；gate 从旧合并权重中切出 | 新增接口适配；不在请求期间反复转换；gamma 不提前加1 |
| `mix` | 原始 `[M,C*H]` 仅添加外层接口维度，调用 `text_gated_residual`；消费 mixed 和 final inject | AscendC 负责 norm/down/SiLU/up/mix/inject；无补零、无裁剪、无旧 HC fallback |
| `_combine_final_kernel` / `combine_final` | 将主干输出按 final inject 权重写回残差：`residual + block_output * injection`；无 injection 时直接相加 | Triton 实现，沿用旧残差写回思路但已修改；不再次执行 `2*sigmoid` 或通道缩放 |
| `combine_and_mix` | 先完成上一主干的残差写回，再调用新 mix | Triton 写回 + AscendC mix；供应商接口不接收上一主干输出，不能用该接口完成这段写回 |
| `vendor_inject_alignment.patch` | 外部供应商 kernel inject 分支的 UB 拷贝长度按32字节对齐 | 单独保留的 AscendC 源码补丁，不会自动应用到本仓库 |

`vllm_ascend/ops/triton/qwen4_exp/hc.py` 没有修改。旧 HC 的 norm/SiLU/gate_mix/combine_norm 等实现仍在基线文件中；新版正常 HC 路径由新类接管，不能把“文件仍存在”当成它们仍被执行。

## 外部算子依赖和本地修复

需要另外安装 `text_gated_residual_release_20261022` 对应的供应商 wheel、目标 SoC 的 OPP 和 Torch bridge。
本分支没有打包 wheel、模型权重、完整供应商源码、native 二进制或原始 trace；仅 checkout 本分支不足以直接运行。

测试使用 `hc_a3_alignfix_v1`，不是完全未修改的供应商原包：

```cpp
// arch22/block/block_epilogue_hc_v1.hpp 的 inject 分支
// 原实现：DataCopy(src, work, nElem);
const uint32_t copyCount = (nElem + 7u) & ~7u;
DataCopy(src, work, copyCount);
```

这里 src/work 是 FP32，8元素即32字节；原槽位已按32字节分配，后续计算仍只处理 nElem 个有效元素。
该修复没有把 M=1 放大为 M=64。补丁和修改前后 SHA256 见同目录文件。
目标编译环境为 Ascend910_9382 / CANN 9.2.0；不同工具链、ABI 或 SoC 必须重新核验，不能直接套用旧缓存。

## 数据流与张量语义

```text
原模型 checkpoint -> 一次准备权重
上一主干输出 -> combine_final(Triton) -> residual[M,10240]
             -> unsqueeze -> [1,M,10240]
             -> TextGatedResidual(AscendC)
             -> mixed[M,2560] + final inject[M,4]
             -> attention / MLP -> 下一次残差写回
```

`10240=4*2560`；外层1是接口维度。TP8 M16 Decode 中，16个请求各贡献1个 token，因此请求 batch 是16。
kernel 内部对 raw FP32 gamma 做 `+1`，适配器不能提前加1；新 inject 是最终权重，不能重复 sigmoid。
供应商 wrapper 自带的输出清零是实际调用成本，性能统计没有扣除。

模型类绑定的生命周期为 worker 进程内、模型构造之前；这是该固定 vLLM 版本的临时继承/绑定适配。
若上游提供稳定注入接口，应改用该接口并移除模块全局类绑定；本分支不声称同进程可同时混用新旧 HC 模型。

## 已有验证与限制

- 曾完成 M1/M64、gate 有无、gamma 有无的8项原生执行诊断；这些只证明执行。
- 修复包的供应商原始默认精度用例2/2通过；模型片段额外诊断18项通过，均不能替代全模型精度。
- 旧 TP4 全模型曾从第28个输出 token 起分歧，尚未完成数值等价证明。
- 最新 TP8/M16 按要求没有重跑精度测试。本次发布仅做源码一致性、语法、差异与敏感信息检查，没有重新运行 NPU 测试。

## 最新 TP8/M16 Decode profiling

两组使用同一环境和权重：TP8/DP1/PP1、V1、EP、16个同步请求、输入128/输出32、MTP及Prefix Cache关闭。
`max_model_len=256`、`max_num_batched_tokens=2048`、`max_num_seqs=32`、`gpu_memory_utilization=0.94`。

```json
{"cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}
```

两次真实预热，实际采31次 M16 Decode 图回放。最终两组采用相同 Level1/AiCoreNone；各8rank设备事件完整后才统计。

| 指标 | baseline | fused | 新版变化 |
| --- | ---: | ---: | ---: |
| 完整 HC 设备 kernel 累计耗时，8rank均值/每卡 | 173.753 ms | 234.954 ms | +35.22% |
| Decode 设备窗口跨度，rank最大值 | 1544.468 ms | 1591.543 ms | +3.05% |
| profiling关闭，整批16请求耗时中位数 | 2.248396 s | 2.271720 s | +1.04% |

完整 HC 包括旧版 norm/combine_norm/down/SiLU/up/gate_mix/combine，或新版 TextGatedResidual/两类输出清零/combine_final。
设备累计耗时不是端到端关键路径；整批计时含 Prefill/Decode/调度，每组仅3次样本且未重验数值等价，不能据此建立稳定的纯 Decode 端到端结论。
图回放后，旧 HC down->SiLU->up->gate 的三段设备间隙平均合计约0.072us，不能沿用旧 M1/eager 的大间隙推断图模式收益。

结果如实记录为当前配置下 HC 设备成本退化，不宣称性能提升或已具备上游合入条件。

## 本地完整证据索引

完整采集保存在本地 `D:/Codex_doc/算子/hc精度测试/HC图模式M16_20261023/TP8_最终profiling`。
新旧分别为 `parsed_profiles/{baseline,fused}/decode/tp{0..7}_ascend_pt/ASCEND_PROFILER_OUTPUT/`。
MindStudio 打开 `trace_view.json`；逐 kernel 数据为 `kernel_details.csv`。
同目录 `对比统计.json`、`逐rank设备与阶段核验.json`、`本地下载校验.json` 保存汇总、完整性与下载核验。
这些本地数据未上传 GitHub。
