---
layout: post
title: "대한민국 독자 AI 파운데이션 모델 4종 비교 — K-EXAONE 2.0 / A.X K2 / Solar Open 2 / Motif 3"
description: "LG AI Research, SK Telecom, Upstage, Motif Technologies가 각각 공개한 파운데이션 모델의 Technical Report를 architecture / pretraining / post-training / serving / license 관점에서 비교한 정리 노트. 벤치마크 점수는 서로 다른 evaluation harness 결과를 nominal-level로 정렬한 cross-report 비교이므로 소수점 단위 순위 해석은 주의가 필요하다."
date: 2026-08-11
category: "Paper Review"
lang: ko
---

2026년 8월 11일을 기준으로 국내 파운데이션 모델 프로젝트의 4개 주요 모델이 모두 공개됐습니다. 네 팀 모두 **대규모 sparse Mixture-of-Experts(MoE)** 라는 큰 방향을 공유하지만, 실제 Technical Report를 열어보면 모델을 키우는 방식, attention 설계, long-context 접근, post-training, serving 최적화, 라이선스 전략에서 꽤 다른 선택이 눈에 띕니다.

이번 글은 네 모델의 Technical Report를 훑으며 정리한 노트입니다. 각 모델의 핵심 기술적 특징을 요약하고, 공개된 benchmark를 최대한 공정하게 비교한 뒤, 마지막에 개인적인 감상과 References를 붙였습니다.

> **평가 해석 시 주의사항**
> 네 모델의 Technical Report 점수는 하나의 통일된 evaluation harness에서 측정된 결과가 아닙니다.
> 따라서 본 보고서의 국내 모델 간 표는 **각 Technical Report에 보고된 점수를 공통 benchmark 이름 기준으로 정렬한 cross-report comparison입니다.**
> 동일 benchmark라도 reasoning effort, max output tokens, sampling, benchmark subset, agent scaffold, tool 사용 여부 등이 다를 수 있으므로 **소수점 단위의 차이를 절대적인 모델 순위로 해석해서는 안 됩니다.**
> 반면 Artificial Analysis Intelligence Index(AAII)는 Artificial Analysis가 자체 harness로 수행한 **독립 평가**이므로, 해당 기관에 평가된 모델끼리는 상대적으로 직접 비교 가치가 높습니다.

---

## 1. Executive Summary

2026년 8월 11일을 기점으로 독자 AI 파운데이션 모델 프로젝트의 주요 4개 모델이 모두 공개되었습니다.

- LG AI Research: K-EXAONE 2.0
- SK Telecom: A.X K2
- Upstage: Solar Open 2
- Motif Technologies: Motif 3

네 모델은 모두 대규모 sparse Mixture-of-Experts (MoE)를 사용하지만, 모델을 키우는 방식, attention 설계, long-context 접근법, Post-Training, serving 최적화, 라이선스 전략은 상당히 다양합니다.

| 항목 | K-EXAONE 2.0 | A.X K2 | Solar Open 2 | Motif 3 Beta |
| --- | --- | --- | --- | --- |
| 개발사 | LG AI Research | SK Telecom | Upstage | Motif Technologies |
| Total Params | **750B** | 688B | 250B | 314B |
| Active Params | 37B | 33B | 15B | **13.2B** |
| Routed Experts | 256 | 256 | 320 | **384** |
| Activated Experts | Top-8 + Shared | Top-8 + Shared | Top-8 + Shared | Top-8 + Shared |
| 초기화 | K-EXAONE Upcycling | **From Scratch** | Solar Open 1의 2.3% transfer | **From Scratch** |
| 주요 Pretraining | +8T CPT | ~8.2T | ~11.9T | ~12.5T |
| Native Context | 256K | 128K | **1M** | 256K |
| 대표 Attention | SWA + Global | MLA + SGA | 1 Softmax : 3 Linear | GDLA |
| Post-training | SFT + RL + Preference | Think-Fusion + Multi-stage RL | 12 Specialists + MOPD | 6 GRPO Specialists + MOPD |
| 대표 시스템 기술 | MTP + DSpark | MXFP8 + NVFP4 | Linear Attention | Muon + MXFP8 |
| License | **Apache 2.0** | **Apache 2.0** | Solar License | Non-commercial Beta |

K-EXAONE 2.0은 기존 K-EXAONE을 236B에서 750B까지 upcycling하면서 기존 학습 투자를 최대한 보존하는 전략을 취했다. A.X K2는 688B 모델을 처음부터 학습하면서 sparse attention, low-precision training, controllable reasoning에 집중했다. Solar Open 2는 15B active compute로 native 1M context와 long-horizon agent를 목표로 하며, Motif 3는 13.2B active compute에 GDLA, mHC, PolyNorm, Muon 등 가장 많은 architectural variation을 적용했다.

---

## 2. 모델별 학습 방법과 기술적 특징

### 2.1 LG AI Research — K-EXAONE 2.0

#### 모델 개요

K-EXAONE 2.0은 **750B total / 37B active**의 fine-grained sparse MoE입니다.

가장 큰 특징은 처음부터 750B 모델을 학습한 것이 아니라, 기존 **K-EXAONE 236B-A23B를 upcycling했다**는 점입니다.

- Layer: 48 → **78**
- Routed Experts: 128 → **256**
- Active Params: 23B → **37B**
- Context: 최대 **256K**

LG는 기존 모델에 들어간 parameter뿐 아니라 compute, data, training recipe까지 버리지 않기 위해 from-scratch training 대신 upcycling을 선택했다고 설명합니다.

#### Architecture

![](/assets/images/korean-foundation-models/image-01.png)

전체 78 layers는:

- 2 Dense Layers
- 76 MoE Layers

로 구성됩니다.

Attention은 기존 K-EXAONE의 **Sliding Window Attention + Global Attention** hybrid 구조를 유지합니다.

MoE에서는:

- 256 Routed Experts
- Top-8 Routed Experts
- Shared Expert 1개

를 사용합니다.

후반부 16개 MoE layer에는 **Clamped SwiGLU**를 적용합니다. 이는 깊은 layer에서 activation이 비정상적으로 커지는 현상을 제어하기 위한 설계입니다.

#### Pretraining

Upcycling 이후 먼저 모델을 안정화하는 healing stage를 수행하고, 이후 **추가 8T tokens의 continual pretraining**을 진행합니다.

이후 별도의 mid-training으로 context를 확장합니다.

| Stage | Training | Context |
| --- | --- | --- |
| Continual Pretraining | +8T | 기본 pretraining |
| Mid-Stage 1 | 400B | 64K |
| Mid-Stage 2 | 400B | 최대 256K |

Mid-training에서는 long-form reasoning, repository-level code, multi-step tool use 비중을 높입니다.

#### Post-training

Technical Report는 K-EXAONE 2.0의 post-training이:

- Supervised Fine-Tuning
- Online Reinforcement Learning
- Preference Learning

을 포함한다고 명시합니다.

다만 구체적인 Section 5에서는 SFT와 Preference Learning이 상세하게 설명되고, online RL의 세부 recipe는 상대적으로 제한적으로 공개됩니다.

SFT에는 약 **350B tokens**가 구성되며, Preference Learning에는 자체 방법인 **GROUPER(Group-wise SimPER)**가 사용됩니다.

따라서 K-EXAONE 2.0의 post-training은 대규모 SFT와 preference/safety alignment가 특히 강조된 구조로 보는 것이 적절합니다.

#### MTP + DSpark

K-EXAONE 2.0은 production inference를 위해 두 speculative decoding path를 제공합니다.

**MTP**

- Backbone과 함께 학습
- Self-drafting head로 사용

**DSpark**

- Final target model이 완성된 뒤 별도로 학습
- Semi-autoregressive 방식으로 draft block 생성

Technical Report의 8×H200, TP8, FP8 환경에서:

- MTP: **1.27~1.77× E2E speedup**
- DSpark: **1.81~2.57× E2E speedup**

을 기록합니다.

따라서 일부 자료에서 언급된 “3~5배 가속”보다는 **최대 약 2.57배 E2E 가속**이 Technical Report 기준 정확한 표현인것으로 보입니다.

#### K-EXAONE 2.0의 포지션

> **기존 대형 모델을 버리지 않고 더 크게 만들면서, long-context·safety·serving까지 안정적으로 가져가는 전략**

이라고 요약할 수 있습니다.

---

## 2.2 SK Telecom — A.X K2

#### 모델 개요

A.X K2는 **688B total / 33B active**의 MoE 모델입니다.

![](/assets/images/korean-foundation-models/image-02.png)

일부 자료에는 692B로 적혀 있지만, 공식 Technical Report와 model card 기준으로는 **688B입니다.**

- 61 Layers
- 256 Routed Experts
- Top-8 Experts
- Shared Expert 1개
- 33B Active Params

K-EXAONE과 달리 **from scratch**로 학습했습니다. (아마도 전 모델이 ….흠)

#### Sparse Gated Attention

A.X K2의 핵심 architecture는 **Sparse Gated Attention(SGA)입니다.**

![](/assets/images/korean-foundation-models/image-03.png)

기본 attention은 MLA를 사용하지만, 긴 context에서는 lightweight indexer가 중요하다고 판단한 **2,048개의 KV position만 선택**한다.

따라서:

- 128K context → 약 **1.6%**
- 256K context → 약 **0.8%**

의 위치에 대해서만 attention을 수행합니다.

A.X K2는 sparse attention을 적용한 상태에서도 LongBench 성능이 거의 변하지 않았다고 보고합니다.

즉 목표는 단순히 context length를 확장하는 것이 아니라:

> **긴 context에서 attention compute 자체를 줄이는 것**

입니다.

#### Gated Norm

A.X K2는 **Gated Norm(GN)**을 사용합니다.

Large Transformer에서는 일부 hidden activation이 다른 값보다 수십~수천 배 커지는 massive activation 문제가 발생할 수 있습니다.

이 현상은 특히 FP8이나 FP4처럼 block 단위 scaling을 사용하는 low-precision format에서 문제가 됩니다.

Gated Norm은 이러한 outlier를 억제해:

- Training stability
- FP8 stability
- FP4 quantization robustness

를 높이는 역할을 합니다.

#### Native MXFP8 Training

A.X K2의 중요한 특징 중 하나는 **pretraining 자체를 MXFP8로 수행했다는 점입니다.**

Forward와 backward에 E4M3 MXFP8을 사용하고, 안정성이 중요한 parameter와 gradient는 FP32로 유지합니다.

또한 공개 checkpoint는 FP8 serving에 바로 사용할 수 있으며, NVFP4 기반 4-bit serving도 지원합니다. 공식 model card는 A.X K2를 Apache 2.0으로 공개하고 research와 commercial use를 모두 허용한다고 설명합니다.

#### Pretraining Curriculum

A.X K2의 pretraining은 약 8.2T tokens입니다

| Stage | Tokens | Context | 목표 |
| --- | --- | --- | --- |
| Stage 1 | 6.4T | 4K | General Knowledge |
| Stage 2 | 1.4T | 4K | Advanced Reasoning |
| Stage 3A | ~0.22T | 32K | Long Context |
| Stage 3B/3C | ~0.14T | 128K | Long Context + SGA |

여기서 중요한 점은 A.X K2의 **native context는 128K**라는 것입니다.

256K는 YaRN scaling을 이용한 inference extension입니다.

따라서 Solar Open 2의 native 1M과 동일한 의미로 비교하면 안 됩니다.

#### Think-Fusion

Post-training은:

**Indexer SFT → Main SFT → Multi-stage On-policy RL**

구조입니다.

Think-Fusion은 동일 prompt에 대해:

- Thinking response
- Non-thinking response

를 모두 학습시켜 하나의 checkpoint에서 reasoning mode를 제어하도록 합니다.

Main SFT 전체 규모는 약 **60.6B tokens**입니다.

#### Multi-stage RL

RL에서는:

- Instruction Following
- Human Preference
- Agentic Tool Use
- Safety

를 함께 최적화합니다.

Optimization에는 **CISPO와 GDPO**를 사용하며 전체적으로 약 100K prompts가 사용됩니다. A.X K2 Technical Report는 RL에 약 100K prompts를 사용했다고 명시합니다.

#### A.X K2의 포지션

> **Math/Korean reasoning + sparse long-context + low-precision deployment + controllable reasoning**

이 가장 명확한 특징입니다.

---

## 2.3 Upstage — Solar Open 2

#### 모델 개요

Solar Open 2는:

- **250B Total**
- **15B Active**
- 48 Layers
- 320 Routed Experts
- Top-8
- Shared Expert 1개
- **1M Context**

의 MoE 모델입니다.

![](/assets/images/korean-foundation-models/image-04.png)

#### Hybrid Linear Attention

Solar Open 2의 attention pattern은:

**Softmax → Linear → Linear → Linear**을 반복합니다.

즉 전체 attention layer의:

- 25%는 Softmax Attention
- **75%는 Linear Attention**

입니다.

Linear attention에는 KDA 계열 구조를 사용하며 positional encoding은 **NoPE**, 즉 explicit positional encoding을 사용하지 않습니다.

#### Negative Eigenvalues

Solar는 KDA recurrent state의 transition 값에 음수를 허용합니다.

기존 recurrent/linear attention이 정보를 주로:

- 유지하거나
- 감소시키는

방향이었다면, Solar Open 2에서는 기존 state를 **반전하거나 적극적으로 수정**하는 dynamics까지 가능하게 만듭니다.

이러한 hybrid attention과 recurrent state 설계로 native **1M context**를 지원합니다.

#### Selective Weight Transfer

Solar Open 2는 Solar Open 1 전체를 upcycling하지 않습니다.

기존 모델에서 architecture가 호환되는 **5.69B parameters, 약 2.3%**만 가져옵니다.

가져오는 부분은:

- Embedding / Output
- Normalization
- 일부 Attention Projection
- Shared Expert

등입니다.

반면 **약 241.7B의 routed expert parameter는 새로 초기화합니**다.

따라서 Solar의 방법은:

> **완전한 from-scratch와 full upcycling의 중간 형태인 partial warm start**

로 볼 수 있습니다.

#### Pretraining

Solar Open 2의 학습 과정은:

**10T General Pretraining → 1T Intensive Pretraining → 865B Length Expansion**

입니다.

총 약 **11.9T tokens**를 학습하고, Length Expansion 단계에서 context를 1M까지 확장합니다.

### Post-training

Solar의 post-training은:

**SFT → Multi-domain RL → 12 Specialist Teachers → MOPD**

구조입니다.

Reasoning, coding, tool use, workspace, search, preference, safety 등 capability별 specialist를 별도로 강화한 뒤 하나의 student model에 통합합니다. Solar Open 2는 12개의 specialist를 MOPD로 통합한다고 명시합니다.

#### MOPD

Multi-teacher On-Policy Distillation에서는 student가 자신의 trajectory를 생성하고, 해당 domain의 teacher가 student가 방문한 state에서 supervision을 제공합니다.

따라서 단순히 teacher의 완성된 response를 SFT하는 것과 달리:

> **student가 실제로 실수하고 drift하는 state에서 teacher에게 배우는 방식**

입니다.

### Solar Open 2의 포지션

> **15B active compute + native 1M context + Korean knowledge work + long-horizon agent**

로 요약할 수 있습니다.

---

## 2.4 Motif Technologies — Motif 3 Beta

#### 모델 개요

Motif 3 Technical Report 기준:

- **314B Total**
- **13.2B Active**
- 53 Layers
- 384 Routed Experts
- Top-8
- Shared Expert 1개
- 256K Context

입니다.

Hugging Face UI에서는 약 315B로 표시되지만 Technical Report architecture 기준으로는 314B입니다.

네 모델 중 active parameter가 가장 적고 expert pool은 가장 큽니다.

![](/assets/images/korean-foundation-models/image-05.png)

#### GDLA

Motif의 핵심 attention architecture는 **Grouped Differential Latent Attention(GDLA)입니다.**

GDLA는:

- Differential Attention
- Grouped Differential Attention
- MLA-style latent KV compression
- Query-dependent output gating

을 결합합니다.

핵심 아이디어는 attention에서 signal과 noise에 공통적으로 나타나는 component를 제거하면서 MLA의 KV cache efficiency를 함께 활용하는 것입니다.

#### mHC

Motif는 conventional residual connection 대신 **modified manifold-constrained hyper-connections(mHC)**를 사용합니다.

일반 Transformer가 하나의 residual stream을 사용하는 것과 달리 여러 residual stream을 mixing하는 방식입니다.

#### Expert-Specific PolyNorm

일반적인 MoE는 expert가 달라도 동일한 activation function을 사용합니다.

Motif는 각 expert가 **자신만의 polynomial activation coefficient**를 학습하게 합니다.

즉 expert specialization을:

- routing
- weights

뿐 아니라 **nonlinearity 자체까지 확장합니**다.

#### Muon

Motif 3는 matrix-shaped parameter의 primary optimizer로 **Muon**을 사용하고:

- Embedding
- LM Head
- Vector parameters

에는 AdamW를 사용합니다.

314B MoE 규모에서 **Muon**을 실제 pretraining에 사용했다는 점도 기술적으로 흥미로운 부분입니다.

#### Pretraining

약 **12.5T tokens**를 학습합니다.

Technical Report에 따르면 NVIDIA Nemotron pretraining collection에서 유지한 데이터가 전체 corpus의 약 70%를 차지하며, 나머지는 자체 web, STEM, code, math, Korean, legal, financial, synthetic data 등으로 구성됩니다.

### Specialist RL

Motif의 post-training은:

**General SFT → 6 GRPO Specialists + 1 SWE SFT Specialist → MOPD**

입니다.

6개의 RL teacher가 총 13개의 verifier domain을 나누어 학습합니다.

모든 verifier reward를 하나의 RL에 섞지 않는 이유는:

- reward variance
- verifier latency
- 서로 다른 failure mode

간 interference를 줄이기 위해라고 합니다.

### Motif 3의 포지션

> **13.2B active compute에서 aggressive architecture + specialist RL로 agent capability를 최대화하는 전략**

으로 볼 수 있습니다.

---

## 3. 네 모델의 기술 전략 한눈에 비교

| 구분 | K-EXAONE 2.0 | A.X K2 | Solar Open 2 | Motif 3 Beta |
| --- | --- | --- | --- | --- |
| Model initialization | Full-model upcycling | From scratch | 2.3% selective transfer | From scratch |
| Attention | SWA + Global | MLA + SGA | 25% Softmax + 75% Linear | GDLA |
| Position | RoPE/SWA hybrid | RoPE + ABF/YaRN | **NoPE** | RoPE 계열 |
| Long Context | 256K native | 128K native / 256K YaRN | **1M native** | 256K native |
| MoE | 256 / Top-8 | 256 / Top-8 | 320 / Top-8 | **384 / Top-8** |
| Stabilization | Clamped SwiGLU | Gated Norm | gated/KDA design | PolyNorm + mHC annealing |
| Low Precision | FP8 serving | **Native MXFP8 training** | B200-scale training | MXFP8 experts |
| Main Optimizer | — | AdamW | — | **Muon + AdamW** |
| Reasoning Control | Thinking / Non-thinking | **Think-Fusion** | reasoning effort | reasoning mode |
| Post-training philosophy | General SFT + preference/RL | Unified multi-task RL | **Specialists → MOPD** | **Specialists → MOPD** |
| Serving-specific feature | **MTP + DSpark** | SGA + NVFP4 | Linear attention / 1M | MTP / fine-grained MoE |

---

## 4. 4-Way Common Benchmark

네 Technical Report에 모두 등장하는 대표 공통 benchmark는 다음 네 개입니다.

> **Cross-report comparison — unified harness가 아님**

| Benchmark | K-EXAONE 2.0 | A.X K2 | Solar Open 2 | Motif 3 |
| --- | --- | --- | --- | --- |
| **GPQA Diamond** | 82.2 | 85.6 | **86.3** | 83.4 |
| **HLE** | 18.3 | 27.8 | 28.8 (w/o tools) | **37.0** |
| **IFBench** | 72.6 | 75.9 | **80.0** | 78.2 |
| **AA-LCR** | 56.2 | 66.0 | 62.3 | **72.3** |

### 4.1 GPQA Diamond

공개 report score 기준:

**Solar 86.3 > A.X K2 85.6 > Motif 83.4 > K-EXAONE 82.2**

Solar와 A.X K2가 상대적으로 강하지만 최신 글로벌 frontier와는 아직 차이가 있습니다.

| 모델 | 공개 형태 | GPQA Diamond | 평가 기준 |
| --- | --- | --- | --- |
| **GPT-5.6 Sol (max)** | Closed | **94.1** | Artificial Analysis |
| **Gemini 3.1 Pro Preview** | Closed | **94.1** | Artificial Analysis |
| **Claude Opus 5 (High Effort)** | Closed | **93.7** | Artificial Analysis |
| **Kimi K3** | Open Weight | **93.5** | Artificial Analysis |
| **GLM-5.2** | Open Weight | **91.2** | 공식 Model Report |
| **DeepSeek V4 Pro** | Open Weight | **90.1** | GLM-5.2 공식 비교표 |
| **Solar Open 2** | Open Weight | **86.3** | Upstage Technical Report |
| **A.X K2** | Open Weight | **85.6** | SKT Technical Report |
| **Motif 3** | Open Weight | **83.4** | Motif Technical Report |
| **K-EXAONE 2.0** | Open Weight | **82.2** | LG Technical Report |

### 4.2 HLE

**Motif 37.0 > Solar 28.8 > A.X K2 27.8 > K-EXAONE 18.3**

Motif가 가장 강한 결과를 보고합니다.

다만 HLE는 report별 subset/setting 차이가 있을 수 있습니다. 예를 들어 LG는 text-only subset, Solar는 without-tools setting을 명시합니다.

| 모델 | 공개 형태 | HLE | 평가 기준 |
| --- | --- | --- | --- |
| **Claude Fable 5 (Max)** | Closed | **53.3** | Artificial Analysis |
| **Claude Opus 5 (Max)** | Closed | **52.6** | Artificial Analysis |
| **Claude Opus 5 (Xhigh)** | Closed | **52.5** | Artificial Analysis |
| Claude Opus 4.8 (Max) | Closed | 49.8 | Kimi K3 공식 비교, No Tools |
| GPT-5.6 Sol (Max) | Closed | 44.5 | Kimi K3 공식 비교, No Tools |
| **Kimi K3 (Max)** | Open Weight | **43.5** | Kimi K3 공식, No Tools |
| **Motif 3** | Open Weight | **37.0** | Motif Technical Report |
| Solar Open 2 | Open Weight | 28.8 | Solar Technical Report |
| A.X K2 | Open Weight | 27.8 | SKT Technical Report |
| K-EXAONE 2.0 | Open Weight | 18.3 | LG Technical Report |

### 4.3 IFBench

**Solar 80.0 > Motif 78.2 > A.X K2 75.9 > K-EXAONE 72.6**

Instruction-following에서 Solar가 가장 높은 공개 점수를 보입니다.

| 모델 | 공개 형태 | IFBench | 평가 기준 |
| --- | --- | --- | --- |
| **Grok 4.3 (Medium)** | Closed | **83.3** | Artificial Analysis |
| **Grok 4.20 0309 (Reasoning)** | Closed | **82.9** | Artificial Analysis |
| **MiniMax-M3** | Open Weight | **82.9** | Artificial Analysis |
| **Solar Open 2** | Open Weight | **80.0** | Solar Technical Report |
| Qwen 3.7 Max | Closed / API | 79.1 | Motif Technical Report 비교표 |
| **Motif 3** | Open Weight | **78.2** | Motif Technical Report |
| DeepSeek V4 Pro | Open Weight | 76.5 | Motif Technical Report 비교표 |
| A.X K2 | Open Weight | 75.9 | SKT Technical Report |
| K-EXAONE 2.0 | Open Weight | 72.6 | LG Technical Report |

## 4.4 AA-LCR

**Motif 72.3 > A.X K2 66.0 > Solar 62.3 > K-EXAONE 56.2**

Long-context reasoning에서는 Motif가 가장 높은 report score를 보입니다.

| 모델 | 공개 형태 | AA-LCR | 평가 기준 |
| --- | --- | --- | --- |
| **GPT-5.2 Codex (Xhigh)** | Closed | **75.7** | Artificial Analysis |
| **GPT-5 (High)** | Closed | **75.6** | Artificial Analysis |
| **GPT-5.1 (High)** | Closed | **75.0** | Artificial Analysis |
| **Kimi K3 (Max)** | Open Weight | **74.7** | Artificial Analysis / Kimi 공식 |
| GPT-5.5 (Xhigh) | Closed | 74.3 | Artificial Analysis / Kimi 공식 |
| GPT-5.6 Sol (Max) | Closed | 73.7 | Artificial Analysis / Kimi 공식 |
| **Motif 3** | Open Weight | **72.3** | Motif Technical Report |
| GLM-5.2 (Max) | Open Weight | 71.3 | Kimi K3 공식 비교 |
| A.X K2 | Open Weight | 66.0 | SKT Technical Report |
| Solar Open 2 | Open Weight | 62.3 | Solar Technical Report |
| K-EXAONE 2.0 | Open Weight | 56.2 | LG Technical Report |

세 개를 요약하면 현재 그림은 다음과 같습니다.

| Benchmark | 독파모 최고 | 점수 | 글로벌 최고권 | 점수 | 인상 |
| --- | --- | --- | --- | --- | --- |
| **HLE** | Motif 3 | 37.0 | Claude Fable 5 | 53.3 | **격차 큼** |
| **IFBench** | Solar Open 2 | 80.0 | Grok 4.3 Medium | 83.3 | **상당히 근접** |
| **AA-LCR** | Motif 3 | 72.3 | GPT-5.2 Codex Xhigh | 75.7 | **매우 근접** |

---

## 5. 3-Way Domain-Specific Benchmark

### 5.1 Math / Korean

#### K-EXAONE 2.0 vs A.X K2 vs Solar Open 2

| Benchmark | K-EXAONE 2.0 | A.X K2 | Solar Open 2 | 글로벌 프론티어 / Reference |
| --- | --- | --- | --- | --- |
| **AIME 2026** | 92.3 | **97.1** | 95.7 | **GLM-5.2 — 99.2** |
| **KMMLU-Pro** | 69.1 | **80.5** | 78.4 | **DeepSeek V4 Pro — 80.5*** |
| **CLIcK** | 84.2 | **91.6** | 90.7 | **DeepSeek V4 Pro — 91.6*** |

AIME 2026은 GLM-5.2 공식 model card 기준 **99.2**가 현재 확인되는 가장 강한 공개 [reference](https://huggingface.co/zai-org/GLM-5.2)입니다. 같은 표에서 GPT-5.5 98.3, Gemini 3.1 Pro 98.2, Qwen3.7-Max 97.0이 보고됩니다.

KMMLU-Pro와 CLIcK는 최신 글로벌 모델을 동일 harness로 폭넓게 평가한 independent leaderboard가 아직 없어, K-EXAONE 2.0 Technical Report의 비교표에서 가장 높은 외부 모델인 **DeepSeek V4 Pro의 80.5 / 91.6**을 reference로 사용하여 표기했습니다.

공개 결과 기준으로 **A.X K2가 독파모 모델중 세 benchmark 모두 가장 높습니다.**

특히 한국어에서는:

- KMMLU-Pro: 80.5
- CLIcK: 91.6

으로 매우 높은 성능을 보입니다.

---

### 5.2 Coding / Agent

#### K-EXAONE 2.0 vs Solar Open 2 vs Motif 3

| Benchmark | K-EXAONE 2.0 | Solar Open 2 | Motif 3 | 글로벌 프론티어 / Reference |
| --- | --- | --- | --- | --- |
| **SWE-bench Verified** | 68.2 | 70.4 | **76.2** | **Claude 4.5 Opus — 76.8*** |
| **τ³-Banking** | 14.2 | 19.6 | **35.3** | **Kimi K3 — 33.4*** |

SWE-bench 공식 leaderboard의 **동일 mini-SWE-agent v2.0.0 harness**에서는 Claude 4.5 Opus High Reasoning이 76.8%로 가장 높고, Gemini 3 Flash와 MiniMax M2.5가 각각 75.8%입니다.

τ³-Banking의 [Artificial Analysis 독립 평가](https://artificialanalysis.ai/evaluations/tau3-banking)에서는 **Kimi K3가 33.4%**, GPT-5.6 Sol Max가 33.0%, Claude Opus 5 High Effort가 32.8%로 현재 최고권입니다.

공개 결과 기준으로 Motif의 agent specialization이 가장 뚜렷합니다.

그런데 Motif 모델의 τ³-Banking 점수가 정말… 신기하네요..? 아마도 Artificial Analysis 독립평가로 하면 더 낮은 점수가 나올것으로 예상됩니다. 

Motif는 추가로 **Terminal-Bench 2.1 = 74.9**를 보고합니다.

> Solar는 `Terminal Bench Hard`, LG와 Motif는 `Terminal-Bench 2.1`을 사용하므로 Terminal score는 같은 행에서 직접 비교하지 않습니다.

---

### 5.3 Scientific Coding

#### K-EXAONE 2.0 vs A.X K2 vs Motif 3

| Benchmark | K-EXAONE 2.0 | A.X K2 | Motif 3 | 글로벌 프론티어 / Reference |
| --- | --- | --- | --- | --- |
| **SciCode** | 40.1 | **41.0** | 40.6 | **Claude Fable 5 — 60.2** |

Artificial Analysis의 현재 SciCode [독립 평가](https://artificialanalysis.ai/evaluations/scicode)에서는 **Claude Fable 5가 60.2%로 가장 높고**, Gemini 3.1 Pro Preview 58.9%, Kimi K3 58.7%가 뒤를 잇습니다.

이 표는 꽤 중요한 결과를 보여줍니다. 독파모 세 모델이 모두 **40~41점 수준**인데 최신 frontier는 약 **60점**이므로, 지금까지 본 benchmark 가운데 **scientific coding은 글로벌 frontier와의 격차가 가장 명확한 영역 중 하나입니다.**

---

## 6. Global Open Frontier와 비교

### 6.1 대표 Reasoning Benchmark

최신 글로벌 open-weight reference로 GLM-5.2와 DeepSeek V4 Pro 등을 놓으면 다음과 같은 그림을 볼 수 있습니다.

| Benchmark | Global Reference | 독파모 공개 최고 |
| --- | --- | --- |
| **AIME 2026** | GLM-5.2 **99.2** | A.X K2 **97.1** |
| **GPQA Diamond** | GLM-5.2 **91.2** | Solar Open 2 **86.3** |
| **KMMLU-Pro** | DeepSeek V4 Pro **80.5*** | A.X K2 **80.5** |

KMMLU-Pro의 DeepSeek V4 Pro 80.5는 LG Technical Report가 제시한 비교값입니다.

GLM-5.2 공식 benchmark table은:

![](/assets/images/korean-foundation-models/image-06.png)

- AIME 2026: **99.2**
- GPQA-Diamond: **91.2 | Kimi K3: 93.5**
- DeepSeek V4 Pro GPQA: **90.1**
- DeepSeek V4 Pro AIME 2026: **94.6**

을 보고합니다.

---

## 7. Artificial Analysis Intelligence Index (AAII)

Technical Report의 자체 benchmark 비교와 별개로, Artificial Analysis Intelligence Index는 **Artificial Analysis가 동일한 자체 methodology로 평가하는 composite index**라는 점에서 중요합니다.

2026년 8월 11일 기준 AAII v4.1은 다음 9개 평가를 포함합니다.

1. GDPval-AA v2
2. τ³-Banking
3. Terminal-Bench 2.1
4. SciCode
5. Humanity's Last Exam
6. GPQA Diamond
7. CritPt
8. AA-Omniscience
9. AA-LCR

### 7.1 현재 확인되는 AAII

| 모델 | AAII v4.1 | 비고 |
| --- | --- | --- |
| **Kimi K3** | **57** | Open weights |
| **GLM-5.2 Max** | **51** | Open weights |
| **DeepSeek V4 Pro Max** | **44** | Open weights |
| **Motif 3 Beta** | **44** | Open weight |
| K-EXAONE 2.0 | — | 평가 미확인 |
| A.X K2 | — | 평가 미확인 |
| Solar Open 2 | — | 평가 미확인 |

---

## 8. 라이선스와 공개성

성능뿐 아니라 실제 기업·연구 활용 관점에서는 model license도 중요한 차이입니다.

| 모델 | License | Weight 공개 | Commercial Use | 주요 조건 |
| --- | --- | --- | --- | --- |
| **K-EXAONE 2.0** | **Apache 2.0** | O | **O** | 표준 Apache 2.0 |
| **A.X K2** | **Apache 2.0** | O | **O** | 표준 Apache 2.0 |
| **Solar Open 2** | Upstage Solar License | O | **O** | derivative model 이름 `Solar...`, `Built with Solar` 표시 |
| **Motif 3** | MIT | O | **O** | 저작권·라이선스 고지 유지 |

---

## 9. Capability별 정리

| Capability | 현재 가장 인상적인 모델 | 이유 |
| --- | --- | --- |
| **Math** | A.X K2 | AIME 2026 97.1 |
| **Korean** | A.X K2 / Solar | KMMLU-Pro·CLIcK / Korean Suite |
| **Scientific Reasoning** | Solar / A.X K2 | GPQA 국내 최고권 |
| **Instruction Following** | Solar | IFBench 80.0 |
| **Long-context Architecture** | Solar | Native 1M |
| **Long-context Reasoning** | Motif | AA-LCR 72.3 |
| **Tool / Agent** | Motif | τ³-Banking 35.3 |
| **Korean Office Work** | Solar | Ko-GDPval 86.8 |
| **Safety** | K-EXAONE | Korean safety pipeline |
| **Low-precision Deployment** | A.X K2 | MXFP8 + NVFP4 |
| **Speculative Decoding** | K-EXAONE | MTP + DSpark |
| **Architecture Novelty** | Motif / Solar | GDLA·PolyNorm·mHC / KDA·NoPE |
| **License Openness** | LG / SKT | Apache 2.0 |
| **Independent AAII** | Motif | AAII 44 |

---

## 개인적인 소감

이번 4개 모델을 정리하면서 가장 흥미로웠던 점은 **모두 MoE를 사용하지만 방향은 꽤 달랐다는 것**입니다.

LG와 SKT는 비교적 안정적인 backbone 위에서 scaling, low-precision, serving 최적화에 집중했고, Upstage와 Motif는 linear attention, NoPE, GDLA, mHC, PolyNorm, Muon처럼 architecture 쪽에서 더 과감한 선택을 했습니다.

성능도 생각보다 많이 올라왔습니다. A.X K2는 수학과 한국어에서 특히 강했고, Solar는 instruction following과 1M context, Motif는 agent/tool use와 long-context reasoning에서 인상적인 결과를 보였습니다. 반면 HLE와 SciCode를 보면 **broad expert reasoning과 scientific coding은 아직 글로벌 frontier와 차이가 꽤 남아 있는 것 같습니다.**

개인적으로 현재 가장 균형이 좋아 보이는 모델은 **A.X K2**, 기술적으로 가장 흥미로운 모델은 **Solar Open 2와 Motif 3**였습니다. K-EXAONE 2.0도 benchmark만 보면 상대적으로 덜 화려하지만, 236B 모델을 750B까지 upcycling하고 MTP·DSpark까지 붙인 점은 실제 서비스 관점에서 의미가 커 보입니다. 

그리고 이번에는 라이선스도 좋았습니다. **K-EXAONE 2.0과 A.X K2는 Apache 2.0, Motif 3는 MIT**로 공개됐습니다. 이제 4개 중 3개 모델을 상당히 자유롭게 활용할 수 있다는 점도 긍정적입니다.

다음 세대에서는 단순 benchmark 상승뿐 아니라, 이런 구조적 선택들이 **실제 latency, serving cost, long-context 활용, agent reliability에서 얼마나 차이를 만드는지**가 더 궁금하네요.

---

## References

**Technical Reports**

- [LG AI Research — *K-EXAONE 2.0* Technical Report (arXiv:2608.04505)](https://arxiv.org/abs/2608.04505)
- [SK Telecom — *A.X K2* Technical Report (GitHub)](https://github.com/SKT-AI/A.X-K2/blob/main/A_X_K2_Tech_Report.pdf)
- [Upstage — *Solar Open 2* Technical Report (arXiv:2607.20062)](https://arxiv.org/abs/2607.20062)
- [Motif Technologies — *Motif 3* Technical Report (arXiv:2608.09119)](https://arxiv.org/abs/2608.09119)

**Evaluation & Benchmarks**

- [Artificial Analysis Intelligence Index (AAII)](https://artificialanalysis.ai/)
- GPQA Diamond — Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*
- HLE — *Humanity's Last Exam*
- IFBench — Instruction Following Bench
- AA-LCR — Artificial Analysis Long Context Reasoning
- SciCode — Scientific coding benchmark
