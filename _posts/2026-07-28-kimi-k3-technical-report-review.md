---
layout: post
title: "[논문 리뷰] Kimi K3 Technical Report — 3T급 MoE를 1M 컨텍스트 에이전트로 만드는 법"
description: "Kimi K3 (2.78T MoE / 104.2B active / 1M context) 테크니컬 리포트 리뷰. 긴 시퀀스(KDA+Gated MLA), 깊이(Attention Residuals), 폭(Stable LatentMoE) 세 축의 확장, 그리고 장기 에이전트 RL·MOPD·MXFP4/8 QAT까지 정리."
date: 2026-07-28
category: "Paper Review"
lang: ko
---


최근 공개된 **Kimi K3**는 전체 **2.78T 파라미터**, 토큰당 **104.2B 활성 파라미터**, 최대 **1M 토큰 컨텍스트**를 갖는 네이티브 멀티모달 MoE 모델입니다.

숫자만 보면 “Kimi K2보다 훨씬 커진 모델”처럼 보이지만, 테크니컬 리포트에서 더 흥미로운 부분은 단순한 크기가 아닙니다. Kimi K3는 모델의 정보 흐름을 다음 세 방향에서 동시에 확장합니다.

- 긴 시퀀스 방향: **KDA + Gated MLA**
- 깊은 레이어 방향: **Attention Residuals**
- 넓은 모델 용량 방향: **Stable LatentMoE**

여기에 장기 에이전트 강화학습, Multi-Teacher On-Policy Distillation(MOPD), MXFP4·MXFP8 기반 QAT, 그리고 KDA와 초대형 MoE를 실제로 학습·서빙하기 위한 시스템 최적화까지 결합했습니다.

이번 글에서는 Kimi K3의 핵심 구조와 학습 방법을 수식과 간단한 예시를 통해 살펴보겠습니다.

---

## 1. Kimi K3 한눈에 보기

| 항목 | Kimi K3 |
|---|---:|
| 전체 파라미터 | 2.78T |
| 활성 파라미터 | 104.2B |
| 레이어 수 | 93 |
| Hidden dimension | 7,168 |
| Routed experts | 896 |
| 토큰당 routed experts | 16 |
| Shared experts | 2 |
| Latent MoE dimension | 3,584 |
| Attention 구성 | 69 KDA + 24 Gated MLA |
| 학습 Context length | 최대 1M |
| Vision encoder | MoonViT-V2, 401M |
| 입력 모달리티 | Text, Image, Video |

MoE 모델에서 **전체 파라미터**와 **활성 파라미터**는 구분해서 봐야 합니다.

Kimi K3는 896개의 routed expert를 가지고 있지만, 토큰 하나를 처리할 때는 그중 16개만 선택합니다. 덕분에 전체 모델 용량은 2.78T까지 늘리면서도 토큰당 계산량은 104.2B 수준으로 제한할 수 있습니다.

다만 나머지 expert weight가 사라지는 것은 아닙니다. 계산에는 일부만 참여하더라도, 전체 2.78T weight를 저장하고 여러 장치에 배치해야 하므로 서빙은 여전히 클러스터 규모의 작업입니다.

---

## 2. 전체 구조: 시퀀스·깊이·너비를 따로 확장한다

![Kimi K3 전체 아키텍처](/assets/images/kimi-k3/k3_figure2_architecture.png)

> **그림 1. Kimi K3 전체 아키텍처.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 2, p.3.

Kimi K3의 한 블록은 대략 다음 패턴을 반복합니다.

```text
KDA + Stable LatentMoE
KDA + Stable LatentMoE
KDA + Stable LatentMoE
Gated MLA + Stable LatentMoE
```

즉 Attention은 `KDA : MLA = 3 : 1` 비율로 구성됩니다. 동시에 오른쪽의 붉은 연결은 현재 레이어가 이전 블록의 표현을 다시 가져오는 **Attention Residuals**를 보여줍니다.

이 구조를 역할별로 나누면 다음과 같습니다.

| 방향 | 해결하려는 문제 | Kimi K3의 방법 |
|---|---|---|
| Sequence | 1M 토큰을 어떻게 효율적으로 처리할까? | KDA + Gated MLA |
| Depth | 93개 레이어를 지나며 초기 정보가 희석되지 않을까? | Attention Residuals |
| Width | 토큰당 계산량을 제한하면서 모델 용량을 키울 수 있을까? | Stable LatentMoE |
| Modality | 텍스트와 시각 정보를 어떻게 하나의 모델에서 처리할까? | MoonViT-V2 + shared backbone |

이제 각각을 조금 더 자세히 살펴보겠습니다.

---

# 3. KDA: 모든 KV를 저장하지 않고, 오차만 고친다

## 3.1 일반 Attention의 장문 문제

일반적인 causal self-attention은 현재 query가 과거의 모든 key와 비교된 뒤, 해당 점수로 value를 가중합합니다.

$$
o_t
=
\operatorname{softmax}
\left(
q_t K_{\le t}^{\top}
\right)
V_{\le t}
$$

이를 생성 단계에서 빠르게 수행하기 위해 과거 토큰의 key와 value를 **KV cache**에 저장합니다.

문제는 문맥이 길어질수록 KV cache도 함께 커진다는 점입니다.

```text
1K tokens  → 1K개 토큰의 KV 저장
100K tokens → 100K개 토큰의 KV 저장
1M tokens   → 1M개 토큰의 KV 저장
```

KDA(Kimi Delta Attention)는 과거 토큰의 KV를 모두 개별적으로 보관하는 대신, key-value 관계를 고정 크기의 상태 행렬에 압축합니다.

$$
S_t\in\mathbb{R}^{d_k\times d_v}
$$

여기서 $S_t$는 “지금까지 본 key와 value의 관계를 압축한 메모리”라고 생각할 수 있습니다.

---

## 3.2 KDA의 핵심 상태 업데이트

KDA의 상태는 다음과 같이 갱신됩니다.

$$
S_t
=
\left(
I-\beta_t k_t k_t^{\top}
\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_t k_t v_t^{\top}
$$

그리고 현재 query로 메모리를 읽습니다.

$$
\tilde{o}_t=S_t^{\top}q_t
$$

기호의 의미는 다음과 같습니다.

| 기호 | 의미 | 직관 |
|---|---|---|
| $S_{t-1}$ | 이전까지의 상태 | 기존 기억 |
| $k_t$ | 현재 key | 정보를 기록할 주소 |
| $v_t$ | 현재 value | 주소에 저장할 내용 |
| $q_t$ | 현재 query | 지금 찾으려는 주소 |
| $\alpha_t$ | 채널별 retention | 기존 기억의 유지율 |
| $\beta_t$ | write strength | 새 정보로 수정하는 강도 |

### `Diag`는 무엇인가?

$\alpha_t$가 다음 벡터라고 해보겠습니다.

$$
\alpha_t=
\begin{bmatrix}
0.9\\
0.5\\
0.2
\end{bmatrix}
$$

그러면

$$
\operatorname{Diag}(\alpha_t)
=
\begin{bmatrix}
0.9&0&0\\
0&0.5&0\\
0&0&0.2
\end{bmatrix}
$$

입니다.

이 행렬을 $S_{t-1}$의 왼쪽에 곱하면 상태 행렬의 각 key channel을 서로 다른 비율로 감쇠시킵니다.

```text
첫 번째 채널 → 90% 유지
두 번째 채널 → 50% 유지
세 번째 채널 → 20% 유지
```

즉 $\alpha_t$는 모든 기억을 같은 비율로 지우는 단일 forget gate가 아니라, **key channel마다 다른 유지율을 적용하는 gate**입니다.

---

## 3.3 Delta Rule 형태로 보면 더 쉽다

먼저 감쇠된 기존 상태를 다음과 같이 두겠습니다.

$$
\bar{S}_{t-1}
=
\operatorname{Diag}(\alpha_t)S_{t-1}
$$

원래 식을 전개하면 다음과 같습니다.

$$
\begin{aligned}
S_t
&=
\bar{S}_{t-1}
-\beta_t k_t k_t^{\top}\bar{S}_{t-1}
+\beta_t k_t v_t^{\top}\\
&=
\bar{S}_{t-1}
+
\beta_t k_t
\left(
v_t-\bar{S}_{t-1}^{\top}k_t
\right)^{\top}
\end{aligned}
$$

괄호 안의 값에 주목할 필요가 있습니다.

$$
v_t-\bar{S}_{t-1}^{\top}k_t
$$

- $\bar{S}_{t-1}^{\top}k_t$: 기존 메모리가 현재 key에서 예측한 value
- $v_t$: 새로 저장해야 하는 실제 value
- 둘의 차이: 기존 기억의 오차

따라서 KDA는 새 value를 무조건 덧붙이지 않습니다.

> **기존 메모리가 틀린 만큼만 현재 key 방향에 수정합니다.**

이것이 이름에 `Delta`가 붙는 이유입니다.

---

## 3.4 간단한 숫자 예시

현재 key가 정규화되어 있고, 기존 메모리가 해당 key에서 다음 값을 예측했다고 가정하겠습니다.

$$
\hat v_t
=
\bar{S}_{t-1}^{\top}k_t
=
0.2
$$

그런데 새로 저장해야 하는 값이 $v_t=0.8$이고, 쓰기 강도가 $\beta_t=0.5$라면 업데이트 이후 같은 key로 읽은 값은

$$
S_t^{\top}k_t
=
(1-\beta_t)\hat v_t+\beta_t v_t
$$

이므로

$$
S_t^{\top}k_t
=
0.5\times0.2+0.5\times0.8
=
0.5
$$

가 됩니다.

- $\beta_t=0$: 기존 값을 그대로 유지
- $\beta_t=1$: 새 값으로 강하게 교체
- $0<\beta_t<1$: 기존 값과 새 값 사이로 이동

문장으로 비유하면 다음과 같습니다.

```text
기존 기억: 프로젝트 담당자 = 철수
새 정보: 프로젝트 담당자 = 영희
β가 클수록 기존 담당자 정보를 영희로 강하게 수정
```

---

## 3.5 Lower-bounded decay는 왜 필요한가?

![KDA lower-bounded decay](/assets/images/kimi-k3/k3_figure3_kda_decay.png)

> **그림 2. KDA의 lower-bounded decay와 chunkwise 계산 변화.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 3, p.5.

KDA는 GPU에서 효율적으로 학습하기 위해 sequence를 chunk로 나눕니다. Chunk 내부는 행렬곱으로 병렬 계산하고, chunk 사이에서만 상태를 전달합니다.

이 과정에서는 여러 retention factor를 곱한 누적 decay가 사용됩니다.

$$
\Gamma=\prod_t \alpha_t
$$

그리고 일부 계산에서 $1/\Gamma$가 필요합니다. $\alpha_t$가 0에 지나치게 가까워지면 $\Gamma$는 매우 작아지고, 역수는 매우 커져 BF16 범위를 넘을 수 있습니다.

Kimi K3는 log-decay를 다음 범위로 제한합니다.

$$
g_t
=
g_{\min}
\operatorname{Sigmoid}
\left(
e^{A}z_t
\right),
\qquad
g_{\min}=-5
$$

$$
\alpha_t=\exp(g_t)
$$

따라서

$$
g_t\in(-5,0),
\qquad
\alpha_t\in(e^{-5},1)
$$

이 됩니다.

기억을 한 번에 무한히 강하게 지울 수 없도록 하한을 둔 것입니다. 보고서는 이를 통해 수치 범위를 안정화하고, 기존에 별도 position-pair 계산이 필요했던 diagonal tile까지 Tensor Core의 dense matrix multiplication으로 처리할 수 있다고 설명합니다.

즉 lower bound는 단순한 학습 안정화 장치가 아니라 **GPU kernel 효율까지 고려한 설계**입니다.

---

# 4. MLA: 압축된 KV로 전체 문맥을 다시 조회한다

KDA는 긴 문맥을 고정 크기 상태에 압축할 수 있지만, 그만큼 정보 간섭이 발생할 수 있습니다.

예를 들어 서로 비슷한 key가 같은 상태 공간에 기록되면 한 정보를 수정할 때 다른 정보도 영향을 받을 수 있습니다. 또한 과거의 특정 문자열이나 세부 토큰을 정확히 다시 꺼내는 작업은 모든 KV를 보관하는 global attention보다 불리할 수 있습니다.

Kimi K3는 이를 보완하기 위해 KDA 세 개마다 Gated MLA 하나를 배치합니다.

```text
KDA → KDA → KDA → Gated MLA
```

MLA(Multi-head Latent Attention)는 각 토큰의 전체 head별 key와 value를 저장하는 대신, 더 작은 latent vector를 저장합니다.

$$
c_t=W_cx_t
$$

Attention 계산 시에는 이 latent에서 head별 content key와 value를 다시 복원합니다.

```text
일반 Multi-Head Attention
→ 각 head의 K와 V를 직접 캐시

MLA
→ 작은 latent c_t를 캐시
→ 필요할 때 K와 V를 복원
```

따라서 MLA는 global token-to-token attention을 유지하면서 일반 MHA보다 KV cache를 줄입니다.

Kimi K3는 MLA 출력에도 입력 의존 gate를 적용합니다.

$$
y_t
=
W_o
\left[
\operatorname{Sigmoid}(W_gx_t)
\odot
\tilde{o}_t
\right]
$$

현재 입력 $x_t$가 global attention 결과의 각 channel을 얼마나 사용할지 결정하는 것입니다.

또한 Kimi K3의 MLA에는 별도의 positional encoding을 적용하지 않는 **NoPE** 설계를 사용합니다. 보고서의 역할 분담은 다음과 같이 이해할 수 있습니다.

```text
KDA
→ 순서, 최근성, 누적 상태를 recurrent update로 표현

MLA
→ 위치 인코딩에 의존하지 않고 전체 문맥에서 content 기반 검색
```

결국 Hybrid Attention의 핵심은 다음과 같습니다.

> **KDA로 대부분의 장문 처리를 저렴하게 수행하고, MLA로 주기적으로 정확한 전역 검색 능력을 보충한다.**

---

# 5. Attention Residuals: 토큰뿐 아니라 레이어 깊이에도 Attention을 적용한다

일반 Transformer의 residual connection은 다음과 같은 형태입니다.

$$
h_{l+1}=h_l+f_l(h_l)
$$

이 구조에서는 이전 모든 정보가 하나의 hidden state에 계속 누적됩니다. 레이어가 깊어질수록 초기 표현은 여러 단계의 변환과 덧셈을 거쳐야 합니다.

Kimi K3는 이를 “깊이 방향의 RNN과 비슷한 병목”으로 보고, **Attention Residuals**(AttnRes)를 사용합니다.

각 레이어 $l$에는 학습 가능한 pseudo-query가 있습니다.

$$
q_l=w_l
$$

Embedding과 이전 레이어 출력은 key이자 value로 사용됩니다.

$$
k_i=v_i=
\begin{cases}
h_{\mathrm{emb}}, & i=0\\
f_i(h_i), & 1\le i\le l-1
\end{cases}
$$

레이어별 attention weight는 다음과 같습니다.

$$
\phi(q_l,k_i)
=
\exp
\left(
q_l^{\top}
\operatorname{RMSNorm}(k_i)
\right)
$$

$$
\alpha_{i\rightarrow l}
=
\frac{
\phi(q_l,k_i)
}{
\sum_{j=0}^{l-1}\phi(q_l,k_j)
}
$$

그리고 현재 레이어의 입력 표현은 이전 표현의 가중합으로 만들어집니다.

$$
h_l
=
\sum_{i=0}^{l-1}
\alpha_{i\rightarrow l}v_i
$$

여기서 pseudo-query는 입력 토큰마다 새로 생성되는 query가 아니라, **각 레이어가 학습하는 깊이 선택용 query**입니다.

예를 들어 어떤 후반 레이어가 다음 가중치를 학습할 수 있습니다.

```text
Embedding 표현       0.10
초기 Block 표현      0.45
중간 Block 표현      0.15
직전 Block 표현      0.30
```

이 레이어는 직전 표현만 전달받는 것이 아니라, 현재 계산에 필요한 초기·중간 표현을 직접 다시 가져옵니다.

모든 레이어 출력을 저장하면 메모리와 pipeline communication 비용이 커지므로 Kimi K3는 레이어를 12개 단위 블록으로 묶는 **Block Attention Residuals**를 사용합니다. 보고서 기준으로 93개 레이어를 8개 블록으로 구성하고, embedding까지 포함해 깊이 방향 source를 관리합니다.

메모리 복잡도는 대략

$$
O(Ld)\rightarrow O(Nd)
$$

로 줄어듭니다.

- $L$: 전체 레이어 수
- $N$: 블록 수
- $d$: hidden dimension

Figure 2의 오른쪽 붉은 연결이 바로 이 깊이 방향의 선택적 정보 흐름을 나타냅니다.

---

# 6. Stable LatentMoE: 896개 Expert를 안정적으로 쓰는 방법

Kimi K3는 토큰마다 896개의 routed expert 중 16개를 선택하고, 모든 토큰이 공통으로 사용하는 shared expert 2개를 둡니다.

일반 MoE에서 선택된 expert가 전체 hidden dimension을 처리하면, 활성 expert 수가 늘수록 통신량과 expert weight traffic도 커집니다.

Kimi K3의 LatentMoE는 routed path를 더 작은 공간으로 줄입니다.

```text
x ∈ R^7168
→ Down projection
→ z ∈ R^3584
→ 896개 중 16개 routed expert
→ Up projection
→ R^7168 복원
```

수식으로는 다음과 같습니다.

$$
u
=
\sum_{i\in T_k(x)}
p_i
E_i^{\mathrm{routed}}
\left(
W^{\downarrow}x
\right)
$$

$$
y
=
\sum_{j=1}^{N_s}
E_j^{\mathrm{shared}}(x)
+
W^{\uparrow}
\operatorname{RMSNorm}(u)
$$

- $T_k(x)$: router가 선택한 top-$k$ expert 집합
- $p_i$: 선택된 expert의 mixture weight
- $N_s=2$: shared expert 수
- Routed expert: $3{,}584$차원 latent space에서 계산
- Shared expert: 전체 $7{,}168$차원에서 공통 변환 수행

하지만 896개 중 16개를 선택하는 극단적인 sparsity는 새로운 문제를 만듭니다.

1. Routed branch의 activation이 폭발할 수 있음
2. 특정 expert에 토큰이 몰릴 수 있음
3. 거의 사용되지 않는 expert가 충분히 학습되지 않을 수 있음

Kimi K3는 이를 RMSNorm, SiTU-GLU, Quantile Balancing으로 해결합니다.

---

## 6.1 RMSNorm: Expert 결과의 크기를 먼저 정리한다

여러 expert 출력의 가중합 $u$는 어떤 expert가 선택되었는지와 routing weight에 따라 scale이 크게 달라질 수 있습니다.

Kimi K3는 이를 바로 up projection하지 않고 먼저 RMSNorm합니다.

$$
W^{\uparrow}\operatorname{RMSNorm}(u)
$$

즉 routed branch가 shared branch와 합쳐지기 전에 크기를 안정화합니다.

---

## 6.2 SiTU-GLU: SwiGLU의 큰 activation을 부드럽게 제한한다

![SiTU-GLU](/assets/images/kimi-k3/k3_figure4_situ_glu.png)

> **그림 3. GLU, SwiGLU, SiTU-GLU의 비교.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 4, p.7.

SwiGLU는 다음과 같이 두 branch를 곱합니다.

$$
\operatorname{SwiGLU}(x)
=
\left[
W_gx\odot
\operatorname{Sigmoid}(W_gx)
\right]
\odot W_ux
$$

문제는 gate branch와 up branch가 모두 큰 값을 가질 수 있다는 점입니다. 두 값이 곱해지면 activation outlier와 저정밀 overflow 위험이 커집니다.

Kimi K3는 다음 soft cap을 사용합니다.

$$
\operatorname{softcap}(z,\beta)
=
\beta\tanh\left(\frac{z}{\beta}\right)
$$

그리고 두 branch를 각각 제한합니다.

$$
\operatorname{SiTU\text{-}GLU}(x)
=
\left[
\beta_1
\tanh
\left(
\frac{W_gx}{\beta_1}
\right)
\odot
\operatorname{Sigmoid}(W_gx)
\right]
\odot
\left[
\beta_2
\tanh
\left(
\frac{W_ux}{\beta_2}
\right)
\right]
$$

Kimi K3의 설정은 다음과 같습니다.

$$
\beta_1=4,
\qquad
\beta_2=25
$$

작은 값에서는

$$
\tanh(z)\approx z
$$

이므로 SwiGLU와 비슷하게 동작합니다. 반면 큰 값에서는 tanh가 포화되면서 출력의 증가를 제한합니다.

```text
작은 activation
→ SwiGLU의 유연한 표현력 유지

매우 큰 activation
→ 부드러운 상한을 적용해 폭발 억제
```

Figure 4에서도 SiTU-GLU는 원점 근처에서는 SwiGLU와 유사하지만, 큰 양수 영역에서는 제한된 값으로 수렴합니다.

---

## 6.3 Quantile Balancing: Expert별 목표 토큰 수를 직접 맞춘다

![Quantile Balancing](/assets/images/kimi-k3/k3_figure5_quantile_balancing.png)

> **그림 4. Quantile Balancing의 동작 예시.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 5, p.8.

Router는 토큰 $x_i$에 대해 expert score를 계산합니다.

$$
s_i
=
\operatorname{Sigmoid}(W_rx_i)
$$

Expert별 balancing bias $b$를 더한 score로 top-$k$ expert를 선택합니다.

$$
T_i
=
\operatorname{argtopk}(s_i+b)
$$

실제 mixture weight는 bias가 없는 원래 score로 정규화합니다.

$$
p_{i,j}
=
\frac{s_{i,j}}
{
\sum_{r\in T_i}s_{i,r}
},
\qquad
j\in T_i
$$

즉 bias는 “어떤 expert를 선택할지”만 조절하고, 선택된 expert 출력의 혼합 비율을 직접 왜곡하지 않습니다.

전체 토큰이 $m$개이고, expert가 $n$개이며, 토큰당 $k$개 expert를 선택한다면 expert당 목표 load는

$$
q=\frac{mk}{n}
$$

입니다.

Figure 5에서는 토큰 8개, expert 4개, 토큰당 expert 1개이므로 목표는

$$
q=\frac{8\times1}{4}=2
$$

입니다.

기존 routing 결과는

$$
(4,3,1,0)
$$

처럼 특정 expert에 몰려 있지만, Quantile Balancing 이후에는

$$
(2,2,2,2)
$$

가 됩니다.

좀 더 수식적으로 보면 각 토큰의 top-$(k+1)$ cutoff를 $\alpha_i^{(t)}$라고 할 때, expert $j$의 다음 bias는 margin 분포의 quantile로 설정됩니다.

$$
\hat b_j^{(t+1)}
=
-
\operatorname{quantile}_{1-k/n}
\left(
s_{:,j}-\alpha^{(t)}
\right)
$$

이후 공통 offset을 제거합니다.

$$
b^{(t+1)}
=
\hat b^{(t+1)}
-
\operatorname{mean}
\left(
\hat b^{(t+1)}
\right)\mathbf{1}
$$

기존처럼 bias를 고정 step만큼 조금씩 올리고 내리는 대신, **현재 global batch의 score 분포에서 목표 load에 맞는 cutoff를 직접 계산**하는 접근입니다.

---

# 7. Native Vision: Vision Encoder도 처음부터 같이 학습한다

Kimi K3는 텍스트 모델을 먼저 완성한 뒤 vision encoder를 사후 연결한 구조가 아닙니다.

```text
텍스트 토큰 ─────────────────┐
                             ├→ Shared LLM Backbone
이미지·비디오 → MoonViT-V2 ──┘
```

텍스트와 visual token을 하나의 next-token prediction objective로 공동 학습합니다.

Kimi K3의 MoonViT-V2는 다음 특징을 가집니다.

- 27-layer Vision Transformer
- 약 401M 파라미터
- 이미지와 비디오가 파라미터 공유
- 이미지 내부 spatial attention
- 비디오 frame 사이 temporal attention
- 2×2 pixel shuffle로 visual token 수 축소

특히 기존의 SigLIP 초기화를 사용하지 않고 MoonViT-V2를 처음부터 학습했다는 점이 흥미롭습니다.

![MoonViT-V2 training stability](/assets/images/kimi-k3/k3_figure6_vision_stability.png)

> **그림 5. SigLIP 초기화와 from-scratch MoonViT-V2의 gradient norm 비교.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 6, p.9.

보고서는 사전학습된 vision encoder를 LLM에 붙일 경우 joint optimization 중 gradient spike가 자주 발생했으며, from-scratch MoonViT-V2가 더 낮고 안정적인 gradient norm을 보였다고 설명합니다.

또한 next-token prediction으로 직접 학습하면 OCR, 문서 구조, UI, 세밀한 텍스트와 같이 language modeling에 필요한 시각적 특징을 목적 함수에 맞춰 형성할 수 있다고 주장합니다.

---

# 8. 1M Context는 한 번에 학습하지 않았다

Kimi K3는 처음부터 1M sequence로 전체 사전학습을 진행하지 않았습니다.

```text
Pre-training
8K → 64K

Cooldown
256K → 1M
```

비용이 큰 초장문 학습을 후반부에 집중하는 progressive curriculum입니다.

긴 문서와 비디오에는 중복, 깨진 파일, binary blob, 잘못 잘린 clip 등이 많기 때문에 별도의 cleaning pipeline도 사용했습니다.

또한 단순히 문서 길이가 길다고 모델이 장거리 정보를 사용하게 되는 것은 아닙니다. 정답과 관련된 정보가 가까이 모여 있다면 긴 문맥을 넣어도 short-context 방식으로 해결할 수 있습니다.

그래서 Kimi K3는 여러 문서와 multimodal subtask를 재배열해, 문맥 앞·중간·뒤의 정보를 함께 모아야만 풀 수 있는 synthetic long-context task를 구성합니다.

---

# 9. 2.5배 Scaling Efficiency는 무엇을 의미할까?

![Kimi K3 scaling efficiency](/assets/images/kimi-k3/k3_figure7_scaling_efficiency.png)

> **그림 6. Kimi K2와 Kimi K3의 fitted scaling-law curve.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 7, p.11.

보고서는 architecture, data, training recipe를 함께 개선한 결과 Kimi K2보다 약 **2.5배 높은 overall scaling efficiency**를 얻었다고 설명합니다.

여기서 2.5배는 “같은 GPU에서 토큰을 2.5배 빨리 생성한다”는 뜻이 아닙니다.

Figure 7이 나타내는 것은 scaling-law curve상에서 비슷한 validation loss에 도달하기 위해 필요한 training FLOPs가 감소했다는 의미입니다.

즉 다음과 같이 해석하는 편이 정확합니다.

> **같은 학습 계산 예산에서 더 낮은 validation loss를 얻거나, 같은 loss에 더 적은 FLOPs로 도달한다.**

다만 보고서는 2.5배 개선을 KDA, AttnRes, LatentMoE 같은 개별 요소별로 완전히 분해한 표를 제공하지 않습니다. 따라서 이를 특정 구성요소 하나의 효과로 해석해서는 안 됩니다.

---

# 10. Post-Training: 하나의 RL이 아니라 9개의 Specialist를 만든다

Kimi K3의 post-training은 다음 세 단계로 진행됩니다.

```text
SFT
→ 도메인 × 추론 강도별 RL
→ Multi-Teacher On-Policy Distillation
```

## 10.1 SFT: RL을 시작할 수 있는 기본 Agent 만들기

SFT 데이터는 단순한 질문과 답변만 포함하지 않습니다.

- Reasoning
- Tool selection
- Tool call
- Observation 해석
- 오류 복구
- 결과 검증
- Long-horizon execution

즉 RL에 들어가기 전에 모델이 에이전트 trajectory의 기본 형식과 행동을 이해하도록 합니다.

---

## 10.2 세 도메인 × 세 Reasoning Effort

RL 도메인은 크게 세 가지입니다.

1. **General**  
   지식, 추론, 검색, 비전, faithfulness, knowledge work

2. **General Agent**  
   Deep research, 장기 비서 작업, writing, professional workflow

3. **Coding Agent**  
   Software engineering, GPU kernel, web development

각 도메인마다 다음 세 추론 강도를 학습합니다.

- low
- high
- max

따라서 총 9개의 specialist policy가 만들어집니다.

$$
3\ \text{domains}
\times
3\ \text{reasoning efforts}
=
9\ \text{specialists}
$$

Reasoning effort는 단순히 system prompt에 “짧게 생각해”라고 쓰는 방식이 아닙니다.

문제 $x$마다 cold-start model로 초기 token budget $b_0(x)$를 추정하고, 실제 사용량 $T(y)$이 다음 한도를 넘으면 reward를 $-1$로 바꿉니다.

$$
T(y)>\tau b_0(x)
$$

- 큰 $\tau$: max effort
- 중간 $\tau$: high effort
- 작은 $\tau$: low effort

General task에서는 주로 thinking token을 세고, Agent task에서는 reasoning뿐 아니라 tool-call argument와 누적 output token까지 포함합니다.

즉 reasoning effort는 **reward를 통해 학습된 행동 특성**입니다.

---

## 10.3 Rollout과 Partial Rollout

Rollout은 모델이 한 문제를 받고 최종 결과를 만들 때까지의 한 번의 전체 실행입니다.

```text
문제 이해
→ 계획 수립
→ Tool call
→ Observation
→ 오류 수정
→ 추가 행동
→ 최종 결과
```

장기 Agent task에서는 rollout 길이가 크게 다릅니다.

```text
Rollout A: 10 steps
Rollout B: 100 steps
Rollout C: 1,000 steps
```

모든 rollout이 끝날 때까지 기다리면 가장 긴 작업이 전체 RL iteration을 막습니다.

Kimi K3의 Partial Rollout은 전체 trajectory 중 일정 비율 $\lambda$가 완료되면 먼저 optimization을 시작합니다.

```text
완료된 rollout
→ 이번 iteration에서 학습

미완료 rollout
→ Pause
→ Sandbox와 상태 저장
→ 다음 iteration에서 Resume
```

Trajectory가 여러 iteration에 걸치면 생성 당시 policy와 현재 policy가 달라지는 staleness가 생깁니다. 보고서는 per-token regularization으로 policy update를 국소적인 범위에 제한해 이를 안정화한다고 설명합니다.

![RL scaling](/assets/images/kimi-k3/k3_figure8_rl_scaling.png)

> **그림 7. RL FLOPs 증가에 따른 평가 점수와 평균 Agent step 변화.**  
> Source: Kimi Team, *Kimi K3 Technical Report*, Figure 8, p.13.

Figure 8에서는 RL FLOPs가 증가하면서 평가 점수뿐 아니라 평균 assistant step도 함께 증가합니다. 즉 더 많은 RL이 단순한 정답률 향상뿐 아니라, 모델이 더 긴 도구 사용 trajectory를 안정적으로 수행하는 방향으로 이어졌다는 것이 보고서의 해석입니다.

---

# 11. MOPD: 9개 Teacher를 Weight가 아닌 Policy Space에서 합친다

9개의 specialist를 각각 서비스하면 요청마다 모델을 선택해야 하고, 모델 9개를 별도로 배포해야 합니다. 한 작업에서 coding과 search처럼 여러 능력을 조합하기도 어렵습니다.

Kimi K3는 **Multi-Teacher On-Policy Distillation(MOPD)**으로 9개의 specialist를 하나의 Student에 통합합니다.

![MOPD pipeline](/assets/images/kimi-k3/mopd_figure1_pipeline.png)

> **그림 8. MOPD의 3단계 학습 파이프라인.**  
> Source: *Multi-Teacher On-Policy Distillation*, Figure 1, p.4.  
> 이 그림은 Kimi K3 보고서가 아니라 MOPD 방법론 논문에서 가져왔습니다.

MOPD는 여러 Teacher의 weight를 평균하는 model merging이 아닙니다.

```text
1. Student가 자신의 policy로 rollout 생성
2. Prompt의 domain에 맞는 Teacher 선택
3. Teacher가 Student trajectory를 prefill
4. 각 token 위치의 Teacher 분포 계산
5. Student가 Teacher 분포에 가까워지도록 업데이트
```

핵심은 **Student가 직접 생성한 trajectory 위에서 학습한다**는 것입니다.

Teacher가 미리 생성한 답변을 고정 데이터로 사용하는 off-policy distillation과 달리, Student가 추론 시 실제로 방문하는 상태에서 Teacher의 지도를 받습니다.

---

## 11.1 Reverse KL 관점

MOPD 방법론의 기본 목표는 Student 분포가 Teacher 분포에 가까워지도록 token-level reverse KL을 최소화하는 것입니다.

$$
\mathcal{L}_{\mathrm{rev\text{-}KL}
}
=
\mathbb{E}_{x,y\sim\pi_\theta}
\left[
\frac{1}{|y|}
\sum_t
\sum_{v\in V}
\pi_\theta(v\mid s_t)
\log
\frac{
\pi_\theta(v\mid s_t)
}{
\pi_{\phi_d}(v\mid s_t)
}
\right]
$$

- $\pi_\theta$: Student policy
- $\pi_{\phi_d}$: 도메인 $d$의 Teacher
- $s_t=(x,y_{<t})$: 현재 prompt와 Student가 생성한 prefix
- $V$: 전체 vocabulary

이 방식은 trajectory 끝의 단일 reward만 사용하는 RL보다 token마다 더 조밀한 신호를 제공합니다.

---

## 11.2 Kimi K3는 Sampled-token Log-ratio를 사용한다

전체 vocabulary 분포를 매 token마다 처리하면 계산과 통신 비용이 큽니다.

Kimi K3는 Student가 실제로 생성한 token $y_t$에 대한 Teacher와 Student의 log-probability 차이를 token-level reward로 사용합니다.

$$
r_{\mathrm{opd}}^d
\left(
y_t\mid e,x,y_{<t}
\right)
=
\operatorname{clip}
\left(
\operatorname{sg}
\left[
\log
\frac{
\pi_{\mathrm{teacher}}^{(d,e)}
\left(
y_t\mid x,y_{<t}
\right)
}{
\pi_\theta
\left(
y_t\mid e,x,y_{<t}
\right)
}
\right],
-R_{\max},
R_{\max}
\right)
$$

- $d$: General, Agent, Coding 중 하나
- $e$: low, high, max 중 하나
- $\operatorname{sg}$: stop-gradient
- $\operatorname{clip}$: 지나치게 큰 신호 제한

로그 비율은 다음과 같이 볼 수 있습니다.

$$
\log
\frac{\pi_T(y_t)}{\pi_S(y_t)}
=
\log\pi_T(y_t)-\log\pi_S(y_t)
$$

### 간단한 예시

Student가 `search`라는 token을 생성했다고 가정하겠습니다.

$$
\pi_S(\text{search})=0.2,
\qquad
\pi_T(\text{search})=0.6
$$

그러면

$$
\log0.6-\log0.2
=
\log3
\approx1.10
$$

입니다.

Teacher가 Student보다 해당 token을 높게 평가하므로 positive signal이 됩니다.

반대로

$$
\pi_T(\text{search})=0.05
$$

라면

$$
\log0.05-\log0.2
=
\log0.25
\approx-1.39
$$

가 되어 negative signal이 됩니다.

Kimi K3 보고서는 top-$k$ logit을 이용한 더 세밀한 distillation도 실험했지만, 해당 설정에서는 수렴 속도나 최종 성능에서 명확한 이점을 확인하지 못했다고 설명합니다.

---

# 12. QAT: 배포할 정밀도를 학습 중부터 경험한다

Kimi K3는 post-training이 끝난 뒤 모델을 단순히 4-bit로 변환하지 않습니다. SFT부터 RL까지 **Quantization-Aware Training(QAT)**을 적용합니다.

## 12.1 PTQ와 QAT의 차이

### PTQ: Post-Training Quantization

```text
BF16 모델 학습 완료
→ 저비트로 변환
→ 발생한 성능 저하를 측정
```

모델은 학습 중 quantization error를 경험하지 못합니다.

### QAT: Quantization-Aware Training

```text
Forward에서 저비트 연산 오차 반영
→ 그 오차를 보정하도록 파라미터 학습
```

모델이 실제 배포 정밀도에 미리 적응합니다.

---

## 12.2 Kimi K3의 정밀도 구성

| 구성요소 | Post-training 정밀도 |
|---|---|
| Routed expert weights | MXFP4 |
| Routed expert input activations | MXFP8 |
| Attention projection | Higher precision |
| Latent MoE projection | Higher precision |
| Shared experts | Higher precision |
| Router | Higher precision |

즉 전체 모델을 모두 4-bit로 처리하는 것은 아닙니다.

파라미터 메모리의 대부분을 차지하는 routed expert weight를 MXFP4로 줄이고, 민감한 공통 모듈은 더 높은 정밀도로 유지합니다.

MX는 microscaling format으로, 작은 값 묶음이 하나의 scale을 공유합니다.

```text
저비트 값 여러 개
+
작은 block별 공유 scale
```

Kimi K3는 RL rollout과 training에도 같은 quantization scheme을 사용합니다.

만약 rollout은 고정밀 모델로 생성하고 실제 배포는 저정밀 모델로 수행한다면 다음 mismatch가 생길 수 있습니다.

```text
RL에서 학습한 policy 분포
≠
실제 serving policy 분포
```

같은 정밀도를 사용하면 모델이 실제 배포 환경에서 보일 행동을 기준으로 강화학습할 수 있습니다.

---

# 13. 시스템 관점에서 이 모델이 흥미로운 이유

Kimi K3의 기술 보고서는 모델 구조만 설명하지 않습니다.

- **FlashKDA**: KDA chunk 내부 계산과 state propagation overlap
- **KDA Context Parallelism**: 긴 sequence를 여러 GPU에 분할
- **MoonEP**: Expert Parallelism에서 dynamic redundant expert로 load 균형
- **External KV cache pool**: 장기 Agent rollout의 prefix를 CPU DRAM으로 관리
- **KDA-aware prefix cache**: KDA state와 MLA KV cache를 함께 복원
- **AgentENV**: 긴 RL task를 pause, resume, fork할 수 있는 MicroVM sandbox
- **Speculative decoding replay**: 거부된 draft token 이후 KDA state를 효율적으로 복원

이 부분에서 Kimi K3의 가장 중요한 특징이 드러납니다.

> **새로운 Attention과 MoE 구조를 제안하는 것에서 끝나지 않고, 이를 3T급·1M context 환경에서 실제로 학습하고 서비스하기 위한 kernel, parallelism, cache, scheduler까지 함께 설계했다.**

---

# 14. 보고서를 읽으며 주의해야 할 점

## 14.1 104B Active가 104B Dense와 같다는 뜻은 아니다

토큰당 계산에는 104.2B 파라미터가 활성화되지만, 전체 2.78T expert weight를 저장하고 분산해야 합니다.

따라서 저장 공간, GPU 메모리, expert communication 관점에서는 훨씬 큰 시스템입니다.

## 14.2 1M Context는 무료 기능이 아니다

KDA state는 context length에 따라 커지지 않지만, MLA KV cache는 여전히 sequence length에 비례합니다. 요청마다 KDA recurrent state도 별도로 유지해야 합니다.

1M context를 지원하는 것과, 여러 사용자가 1M context를 동시에 저지연으로 사용하는 것은 다른 문제입니다.

## 14.3 2.5배는 Serving Speed가 아니다

보고서의 2.5배는 scaling-law상 training efficiency 개선입니다. GPU 한 장당 생성 속도가 2.5배 빨라졌다는 의미가 아닙니다.

## 14.4 Benchmark에는 Harness 영향이 크다

Agent benchmark는 모델 자체뿐 아니라 다음 요소에 영향을 받습니다.

- System prompt
- Tool schema
- Retry policy
- Context management
- Sandbox
- Termination rule
- Reasoning effort

따라서 실제 도입 검토에서는 같은 harness와 같은 업무 task로 다시 평가해야 합니다.

## 14.5 재현에 필요한 핵심 정보가 모두 공개된 것은 아니다

보고서는 다음 정보를 구체적으로 공개하지 않습니다.

- 총 사전학습 token 수
- 데이터 도메인별 정확한 비율
- 언어별 데이터 비율
- Peak learning rate와 global batch size
- 전체 training FLOPs
- 사용한 GPU 수와 학습 기간
- RL 전체 rollout budget

구조와 시스템 철학은 상세하지만, 동일한 모델을 처음부터 재현할 정도의 recipe가 모두 공개된 것은 아닙니다.

---

# 15. 마무리

Kimi K3를 단순히 “2.8T 파라미터의 더 큰 MoE”라고만 설명하면 핵심을 놓치게 됩니다.

이 모델의 핵심은 다음 요소들의 결합입니다.

```text
KDA–MLA Hybrid Attention
+
Attention Residuals
+
Stable LatentMoE
+
Native Multimodal Pre-training
+
Progressive 1M Context Extension
+
Domain/Effort Specialized RL
+
Multi-Teacher On-Policy Distillation
+
MXFP4/MXFP8 QAT
+
Training·Serving Infrastructure Co-design
```

KDA는 긴 문맥을 고정 크기 상태로 압축하고, MLA는 주기적으로 global retrieval을 복원합니다. Attention Residuals는 깊이 방향 정보 병목을 줄이며, Stable LatentMoE는 896개의 expert를 안정적으로 활용합니다.

Post-training에서는 하나의 mixed RL을 수행하는 대신 도메인과 reasoning effort별로 9개의 specialist를 만들고, MOPD를 통해 하나의 policy로 통합합니다. 동시에 QAT를 SFT와 RL 전반에 적용해 실제 저정밀 serving 환경까지 학습 과정에 포함합니다.

결국 Kimi K3는 하나의 새로운 모듈보다, **초대형 장문 멀티모달 에이전트를 실제로 만들기 위한 전 계층 공동 설계**에 더 가까운 프로젝트입니다.

---

# 참고 자료

1. Kimi Team, **Kimi K3: Open Frontier Intelligence — Technical Report of Kimi K3**  
   https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf

2. Moonshot AI, **Kimi K3 Model Card**  
   https://huggingface.co/moonshotai/Kimi-K3

3. **Multi-Teacher On-Policy Distillation**  
   https://arxiv.org/abs/2606.30406

