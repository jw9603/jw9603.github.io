---
layout: post
title: "[논문 리뷰] W_K, W_V is (Linearly) All You Need: On the Necessity of the QKV Weight Triplet in Self-Attention Transformers"
description: "Self-Attention에서 W_Q는 Multi-Head Attention의 GL(d) symmetry로 인해 Identity로 축소 가능하며, Query pathway에 진짜 표현력을 더하려면 선형 W_Q가 아니라 비선형 Residual Query가 필요하다는 ICML 2026 Workshop 논문 리뷰"
date: 2026-07-06
category: "Paper Review"
lang: ko
---

# Transformer에서 정말 (W_Q)는 필요한가?

## *WK, WV is (Linearly) All You Need* 논문 리뷰

Transformer의 Self-Attention은 보통 Query, Key, Value 세 개의 projection을 사용한다.

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

우리는 너무 자연스럽게 ($W_Q$), ($W_K$), ($W_V$)를 모두 학습 가능한 파라미터로 둔다. 하지만 이 논문은 꽤 도발적인 질문을 던진다.

> **정말 (W_Q)를 학습해야 하는가?**

논문의 답은 상당히 강하다.

> **선형 영역에서는 (W_Q)는 본질적으로 redundant하다. 더 많은 표현력을 원한다면, Query 경로는 선형이 아니라 비선형이어야 한다.**

이 논문은 기존의 두 흐름, 즉 ($W_Q$)를 Identity로 둘 수 있다는 연구와 Query projection에 비선형성을 넣는 연구를 하나의 weight-space symmetry 관점에서 통합한 ICML 2026 Workshop 논문이다. 논문은 Multi-Head Attention이 가지는 ($GL(d)$) 대칭성을 출발점으로 삼아, ($W_Q = I$) 축소와 Residual Nonlinear Query의 필요성을 이론과 실험으로 연결한다.

---

# 1. 핵심 아이디어: Attention은 (X) 자체가 아니라 세 개의 곱만 본다.

이 논문의 출발점은 단순하다.

Multi-Head Attention은 입력 ($X$)를 직접 보는 것이 아니라, 다음 세 개의 곱만 사용한다.

$$
XW_Q,\quad XW_K,\quad XW_V
$$

즉 Attention 입장에서는 ($X$), ($W_Q$), ($W_K$), ($W_V$)가 각각 무엇인지가 중요한 것이 아니라, 최종적으로 만들어지는 ($Q$), ($K$), ($V$)가 중요하다.

이 사실 때문에 다음과 같은 변환이 가능하다.

$$
(X, W_Q, W_K, W_V) \mapsto (X\Theta,\Theta^{-1}W_Q,\Theta^{-1}W_K,\Theta^{-1}W_V)
$$

여기서 $\Theta$는 $GL(d)$, 즉 역행렬이 존재하는 $d \times d$ 행렬이다.

이 변환을 적용해도 실제 Query는 변하지 않는다.

$$
(X\Theta)(\Theta^{-1}W_Q)=XW_Q
$$

Key와 Value도 마찬가지다.

$$
(X\Theta)(\Theta^{-1}W_K)=XW_K
$$

$$
(X\Theta)(\Theta^{-1}W_V)=XW_V
$$

따라서 Attention의 출력은 동일하다. 논문은 이것을 **Reparametrisation Symmetry**라고 부른다. 더 직관적으로 말하면, Attention은 residual stream의 좌표계가 어떻게 잡혀 있는지에 대해 불변이다. 좌표계를 바꾸더라도 그에 맞게 $W_Q, W_K, W_V$를 반대로 바꾸면 같은 계산을 하게 된다.

---

# 2. 왜 $W_Q = I$가 가능한가?

이제 논문의 가장 중요한 트릭이 나온다.

위 식에서

$\Theta = W_Q$

로 두면,

$$
\Theta^{-1}W_Q = W_Q^{-1}W_Q = I
$$

가 된다.

즉, 원래의 $W_Q$를 Identity matrix로 바꿀 수 있다. 대신 입력은 $XW_Q$로 바뀌고, $W_K, W_V$는 각각 $W_Q^{-1}W_K$, $W_Q^{-1}W_V$로 재매개변수화된다.

결과적으로 다음 등식이 성립한다.

$$
\text{MHA}(X,W_Q,W_K,W_V,W_O) = \text{MHA}(XW_Q,I,W_Q^{-1}W_K,W_Q^{-1}W_V,W_O)
$$

이 말은 꽤 강하다.

> **선형 (W_Q)는 Identity와 같은 symmetry orbit 위에 있다.**

즉, $W_Q$를 학습한다고 해서 새로운 함수 표현력이 생기는 것이 아니라, 같은 Attention 계산을 다른 좌표계로 표현하는 것에 가깝다.

논문은 여기서 $W_Q$가 거의 항상 invertible하다는 점도 짚는다. 일반적인 연속분포 초기화에서는 특이행렬이 나올 확률이 0이므로, $W_Q \in GL(d)$라고 거의 확실하게 볼 수 있다. 그래서 $\Theta=W_Q$라는 선택이 수학적으로 가능해진다.

---

# 3. $W_Q$를 제거하면 파라미터를 얼마나 줄일 수 있나?

표준 Multi-Head Attention에는 보통 네 개의 큰 projection이 있다.

$$
W_Q,\quad W_K,\quad W_V,\quad W_O
$$

각각이 대략 $d^2$개의 파라미터를 가진다. 따라서 $W_Q$를 Identity로 고정하면 Attention projection 파라미터 중 약 25%를 줄일 수 있다.

논문은 더 나아가 GQA(Grouped Query Attention)에서는 이 절감 비율이 더 커질 수 있다고 설명한다. $W_K$, $W_V$가 여러 Query head에서 공유되기 때문에, 상대적으로 $W_Q$가 차지하는 비중이 커지기 때문이다. 논문에서는 LLaMA-style GQA에서는 attention parameter의 약 40%, MQA limit에서는 50%에 가까운 절감이 가능하다고 설명한다.

다만 이론적으로 한 layer에서 $W_Q$를 Identity로 만드는 것은 쉽지만, deep network에서는 한 layer의 basis change가 다음 layer의 입력에도 영향을 준다. 그래서 논문은 Appendix에서 multi-layer propagation 조건을 분석한다. 핵심은 단순한 normalisation-free 모델에서는 정확한 대칭성을 증명하고, LayerNorm이 있는 실제 모델에서는 고차원에서 근사적으로 이 대칭성이 유지될 수 있음을 보인 뒤, 실제 GPT-style 모델에서 실험적으로 검증하는 구조다.

---

# 4. 실험 1: $W_Q = I$는 정말 성능을 유지하는가?

논문은 GPT-3-small/NanoGPT-style 모델을 사용한다. 설정은 12 layer, 12 head, $d=768$, context length 1024, GPT-2 BPE tokenizer, GELU, LayerNorm, FlashAttention 기반이다. 학습은 OpenWebText에서 총 약 29.5B tokens, 즉 해당 규모에서 약 12× Chinchilla-optimal에 해당하는 heavy overtraining regime으로 진행한다.

비교한 모델은 다음과 같다.

| 설정 | Non-embedding params | Validation loss |
| --- | --- | --- |
| Baseline 124M | 84.95M | 2.944 |
| Baseline 117M, MLP 3.5× | 77.88M | 2.955 |
| Baseline 118M, (d=744) | 79.73M | 2.956 |
| (W_Q=I) 117M | 77.88M | 2.944 |
| (W_Q=I) + MLP 4.5× | 84.95M | 2.929 |
| MLP 4.75×, +12.5% non-emb params | 95.57M | 2.917 |
| Residual Nonlinear Query | 84.97M | 2.874 |

여기서 가장 중요한 결과는 두 가지다.

첫째, $W_Q=I$ 모델은 non-embedding parameter를 약 8% 줄였음에도 124M baseline과 같은 validation loss를 기록했다. 즉 이 실험에서는 $W_Q$의 $d^2$ 파라미터가 validation log-loss에 실질적으로 기여하지 않았다.

둘째, $W_Q$를 제거해서 절약한 파라미터를 MLP에 재투자하면 baseline보다 더 좋은 성능을 얻었다. 이는 이 규모의 모델에서는 linear query projection보다 feed-forward sublayer 쪽이 더 중요한 capacity bottleneck일 수 있음을 시사한다.

이 결과는 논문의 첫 번째 주장을 뒷받침한다.

> **MLP를 줄이거나 $d_{\text{model}}$을 줄이면 성능이 떨어지지만, $W_Q$를 제거하는 것은 거의 손실이 없다.**

---

# 5. 그러면 왜 비선형 Query가 필요한가?

여기서 논문의 두 번째 주장이 나온다.

선형 $W_Q$는 Identity와 같은 orbit에 있다. 따라서 아무리 선형 $W_Q$를 학습해도, 이는 결국 basis change로 흡수될 수 있다. 그렇다면 Query pathway에 진짜 새로운 표현력을 주려면 무엇이 필요할까?

논문의 답은 **비선형성**이다.

$$
Q(X)=\frac12(X+f_\theta(X))
$$

여기서 $f_\theta$는 bottleneck MLP다.

$$
f_\theta(X)=LN(GELU(RMSNorm(X)W_1)W_2)
$$

$$
W_1\in\mathbb{R}^{d\times r},\quad W_2\in\mathbb{R}^{r\times d},\quad r=d/2
$$

이 구조의 중요한 점은 파라미터 수가 기존 $W_Q$와 거의 같다는 것이다. 즉, 모델을 크게 만든 것이 아니라 기존 $W_Q$가 차지하던 파라미터 budget을 비선형 Query로 바꾼다.

왜 residual 형태일까?

$$
\frac12(X+f_\theta(X))
$$

여기서 $X$는 $W_Q=I$ prior를 유지한다. 즉, 처음부터 완전히 새로운 Query를 만드는 것이 아니라, Identity Query를 기본값으로 두고 그 위에 비선형 보정을 더한다. 이는 gradient flow 측면에서도 유리하다.

또한 이 변경은 pre-attention query pathway에만 들어간다. Softmax attention kernel 자체는 바꾸지 않기 때문에 FlashAttention과 같은 기존 최적화된 attention 구현을 그대로 사용할 수 있다.

---

# 6. 실험 2: Residual Nonlinear Query는 얼마나 좋아지는가?

가장 좋은 Residual Nonlinear Query 설정은 validation loss를 2.9441에서 2.8736으로 낮췄다. 이는 log-loss 기준 2.40% 개선, perplexity 기준 6.81% 감소에 해당한다. 흥미로운 점은 추가 파라미터가 거의 없다는 것이다.

비교를 위해 MLP hidden width를 4d에서 4.75d로 늘려 non-embedding parameter를 12.5% 증가시킨 모델은 log-loss를 0.94% 개선하는 데 그쳤다. 즉 이 실험에서는 단순히 모델을 크게 키우는 것보다, (W_Q) 자리에 비선형 Query를 넣는 것이 훨씬 효율적이었다.

이 결과가 논문의 가장 강한 empirical claim이다.

> **같은 파라미터 수라면, 선형 (W_Q)보다 Residual Nonlinear Query가 훨씬 낫다.**

논문은 이 결과를 단순한 최적화 효과가 아니라 Query function class의 표현력 증가로 해석한다.

---

# 7. QK-Norm 논의: 그냥 Learning Rate가 커져서 좋아진 것 아닌가?

이 논문에서 흥미로운 방어 논리 중 하나는 QK-Norm과의 비교다.

Residual Nonlinear Query는 baseline보다 훨씬 높은 learning rate를 견딘다. 최고 설정에서는 baseline의 약 6배에 해당하는 peak learning rate $3.6\times10^{-3}$까지 안정적으로 학습된다. 그러면 자연스럽게 이런 반론이 가능하다.

> “성능 향상은 구조적 표현력 때문이 아니라, 그냥 더 큰 learning rate를 쓸 수 있어서 생긴 것 아닌가?”

논문은 여기서 QK-Norm을 control로 사용한다. QK-Norm도 attention logit의 성장을 억제해서 learning-rate horizon을 넓혀준다. 하지만 기존 연구에 따르면 최적으로 튜닝된 learning rate에서의 최종 loss는 거의 변하지 않는다. 즉, 더 넓은 learning-rate horizon 자체는 capacity 증가의 증거가 아니다.

반면 Residual Nonlinear Query는 learning-rate horizon을 넓히는 동시에, best-tuned log-loss를 2.40% 개선한다. 그래서 저자들은 이것이 단순한 optimization side-effect가 아니라 query function class의 genuine expressivity 증가라고 주장한다.

이 부분은 개인적으로도 중요하게 느껴진다. 실제로 현대적인 training stack에서는 QK-Norm, Muon optimizer, RMSNorm 등 여러 안정화 장치가 들어가면서 원래 논문에서 보였던 최적화적 이점이 baseline 쪽으로 흡수될 가능성이 있다. 따라서 후속 연구에서는 “비선형 Query의 이득이 어떤 training stack에서 유지되는가?”를 별도로 확인해야 한다.

---

# 8. Symmetry-breaking 관점

논문의 Discussion에서 가장 중요한 해석은 **symmetry breaking**이다.

$W_Q=I$ 모델은 원래 parameterization의 symmetry orbit에서 하나의 canonical representative를 고른 것이다. 즉, 여러 동등한 표현 중 하나를 고른 것에 가깝다.

반면 Residual Nonlinear Query는 이 대칭성을 깨뜨린다.

$$
X \mapsto \frac12(X+f_\theta(X))
$$

이 map은 $GL(d)$ basis change와 commute하지 않는다. 쉽게 말해, 선형 $W_Q$처럼 $\Theta$와 $\Theta^{-1}$로 깔끔하게 상쇄되지 않는다. 따라서 Lemma 2.1의 reparametrisation argument가 더 이상 적용되지 않는다. 이때 Query projection은 단순한 좌표계 선택이 아니라 실제 expressive content를 갖게 된다.

이 논문을 관통하는 핵심은 바로 이 지점이다.

| Query 구조 | GL(d)로 흡수 가능? | 표현력 |
| --- | --- | --- |
| Linear $W_Q$ | 가능 | Identity와 같은 orbit |
| $W_Q=I$ | canonical representative | 선형 Query의 기본형 |
| $X+f_\theta(X)$ | 불가능 | 새로운 함수 클래스 |

즉 논문의 주장은 단순히 “비선형을 넣으니 성능이 좋아졌다”가 아니다.

> **선형 $W_Q$는 대칭성 때문에 redundant하고, 이 대칭성을 깨는 방식으로만 Query pathway에 진짜 capacity를 줄 수 있다.**

---

# 9. Skip Connection과 MLP 표현력

이 논문은 $W_Q$와 Nonlinear Query만 다루는 것처럼 보이지만, Appendix에서는 MLP의 skip connection에 대한 별도 분석도 포함한다. 논문은 현대 activation에서 skip이 있는 MLP와 skip이 없는 MLP가 일반적으로 서로 다른 함수 클래스를 이룬다고 주장한다. 특히 ReLU, SwiGLU, GeGLU 같은 activation에서는 skip absorption이 원칙적으로 불가능하거나, ReLU/GELU에서는 매우 특수한 경우에만 가능하다고 설명한다.

이 부분은 왜 절약한 $W_Q$ 파라미터를 선형 Query에 다시 쓰는 것보다 MLP 쪽에 쓰는 것이 더 유리한지를 설명하는 보조 이론으로 기능한다.

간단히 말하면, 선형 Query는 $GL(d)$ orbit 안에서 redundant하지만, MLP와 residual skip은 실제 함수 클래스의 경계를 바꾼다. 따라서 같은 파라미터를 쓴다면 선형 Query보다 MLP나 비선형 Query에 쓰는 편이 더 의미 있다.

---

# 10. 한계와 후속 연구

논문은 스스로의 한계도 명확히 인정한다.

첫째, 실험은 하나의 architecture, 하나의 seed, matched batches로 진행되었다. 이는 hyperparameter tuning artifact를 줄이기 위한 보수적인 설계지만, 동시에 multi-seed validation과 더 큰 규모의 실험이 필요하다는 뜻이기도 하다.

둘째, 실험 규모는 124M 수준이다. 논문은 약 12× Chinchilla overtraining 환경에서도 이득이 유지된다는 점을 강조하지만, top-tier main track 수준의 주장을 위해서는 GPT-2 Medium, 더 나아가 1B~3B급 모델에서의 검증이 필요하다.

셋째, downstream task에서 $W_Q=I$ parity와 Residual Nonlinear Query gain이 유지되는지도 아직 확인해야 한다.

넷째, 논문은 “Query-side capacity는 비선형이어야 한다”는 구조적 주장만 한다. 어떤 activation, bottleneck width, depth, normalization 조합이 최적인지는 아직 열려 있다.

논문이 제안하는 자연스러운 후속 방향은 다음과 같다.

- $W_Q=I$와 더 큰 nonlinear query를 결합하기
- $W_K, W_V, W_O$에도 residual nonlinearity를 확장하기
- GQA, RoPE, QK-Norm, MoE 환경에서 검증하기
- GPT-2/3-style setup을 넘어 Gemma, Qwen 등 현대적 architecture로 확장하기
- 단순 validation loss뿐 아니라 downstream generalization 평가하기
- custom kernel이나 projection fusion으로 efficiency 개선하기

---

# 11. 개인적으로 생각하는 이 논문의 의의

이 논문의 가장 큰 의의는 Residual Nonlinear Query 자체보다도, Transformer attention을 **좌표계와 대칭성의 관점에서 다시 보게 만든다**는 점이라고 생각한다.

기존에는 $W_Q, W_K, W_V$를 각각 독립적인 학습 파라미터로 바라봤다. 하지만 이 논문은 Attention이 $XW_Q, XW_K, XW_V$라는 세 곱에만 의존한다는 사실을 이용해, $W_Q$가 사실상 basis choice에 가까운 역할을 할 수 있음을 보인다.

이 관점은 단순한 parameter reduction 이상의 의미가 있다.

- 어떤 projection이 진짜 표현력을 가지는가?
- 어떤 projection은 단순한 reparametrisation인가?
- linear capacity와 nonlinear capacity는 어디서 갈라지는가?
- attention block에서 parameter budget을 어디에 쓰는 것이 가장 효율적인가?

이런 질문으로 이어진다.

특히 “선형 projection은 orbit 안에서 redundant하지만, 비선형 query는 symmetry를 깨면서 새로운 함수 클래스를 만든다”는 주장은 Transformer architecture search에 꽤 흥미로운 방향을 제시한다.

---

# 12. 정리

이 논문을 한 문장으로 요약하면 다음과 같다.

> **Multi-Head Attention의 $GL(d)$ reparametrisation symmetry 때문에 선형 $W_Q$는 Identity로 축소될 수 있으며, Query pathway에 진짜 표현력을 추가하려면 선형 projection이 아니라 symmetry를 깨는 비선형 구조가 필요하다.**

핵심 결과는 다음과 같다.

1. Multi-Head Attention은 $XW_Q, XW_K, XW_V$에만 의존하므로 $GL(d)$ symmetry를 가진다.
2. $\Theta=W_Q$를 선택하면 $W_Q$를 Identity로 만들 수 있다.
3. $W_Q=I$는 8% fewer non-embedding parameters에서도 124M baseline과 같은 validation loss를 보였다.
4. 절약한 파라미터를 MLP에 재투자하면 baseline보다 좋아졌다.
5. $Q(X)=\frac12(X+f_\theta(X))$ 형태의 Residual Nonlinear Query는 동일 파라미터 수에서 log-loss를 2.40% 개선했다.
6. 이 개선은 단순히 learning rate 안정성 때문이 아니라, Query function class의 표현력 증가로 해석된다.
7. 하지만 더 큰 모델, multi-seed, downstream task, 현대 architecture에서의 검증은 아직 필요하다.

개인적으로 이 논문은 “Transformer에서 어떤 선형 파라미터가 진짜 필요한가?”라는 질문을 상당히 우아하게 제기한 연구라고 생각한다. 특히 $W_Q=I$와 Residual Nonlinear Query를 하나의 symmetry-breaking narrative로 연결한 점이 강하다.

다만 현재 결과는 아직 124M 규모의 GPT-style 모델에 머물러 있다. 따라서 이 아이디어가 정말 강한 architecture-level contribution이 되려면, 앞으로는 GPT-2 Medium 이상 규모의 검증, Chinchilla-parity regime에서의 비교, QK-Norm/Muon과 같은 현대 training stack에서의 재현성 확인, 그리고 Gemma/Qwen 계열 모델로의 확장이 필요하다.

그 검증까지 성공한다면, 이 연구는 단순한 workshop 아이디어를 넘어 Transformer architecture를 다시 설계하는 꽤 강한 메인 트랙 논문으로 발전할 수 있을 것 같다. 개인적으로도 이 후속 연구에 참여하고 있는 만큼, 앞으로의 실험 결과가 더욱 기대된다.