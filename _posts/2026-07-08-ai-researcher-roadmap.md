---
layout: post
title: "지금 나는 어디쯤이고, 어디로 가고 있는가 — AI Researcher로서의 로드맵"
description: "회사 일과 개인 연구를 병행하며 세계 최고 수준의 AI·NLP Research Scientist로 성장하기 위한 현재 위치, 연구 방향, 그리고 앞으로의 학습 계획을 스냅샷으로 정리한 글"
date: 2026-07-08
category: "Reflections"
lang: ko
---

지난 몇 달을 돌아보면 회사 일 외에도 꽤 많은 일을 벌여두고 있었다.

EleutherAI 커뮤니티에서 만난 연구자들과 Transformer 아키텍처를 연구하고, Pseudo-Lab에서 진행한 Stanford CS336 스터디에 참여하며 LLM 시스템을 공부했고, Hugging Face Transformers에 Pull Request를 기여했다. 개인 블로그에는 관심 있는 논문을 리뷰하고 구현 과정과 연구 노트를 꾸준히 기록했으며, 새로운 연구 아이디어도 하나씩 실험해 보고 있다.

각각은 분명한 이유가 있어서 시작한 일이었다. 하지만 문득 이런 생각이 들었다.

> **나는 지금 어디쯤 와 있고, 앞으로 어디를 향해 가고 있는 걸까?**

논문 하나를 읽고, 실험 하나를 끝내는 것은 비교적 명확하다. 하지만 커리어는 훨씬 긴 시간 위에서 만들어진다. 지금의 선택들이 결국 어떤 연구자로 이어질지, 한 번쯤은 스스로 정리해 둘 필요가 있다고 생각했다.

이 글은 스스로에게 쓰는 기록에 가깝다. 다만 공개로 남기는 이유는, 시간이 지난 뒤에도 지금의 생각을 다시 돌아볼 수 있는 기준을 만들고 싶었기 때문이다.

---

# 지금의 나

현재 롯데이노베이트 AI Tech LAB Language AI Team에서 AI Research Engineer로 일하고 있다.

회사에서는 LLM 기반 AI 서비스와 다양한 PoC를 수행하고 있으며, 개인 시간에는 보다 근본적인 AI 연구를 이어가고 있다.

학부에서는 전자공학을 전공했고, 이후 성균관대학교 AI 대학원에서 NLP를 연구하며 석사 과정을 마쳤다. 운 좋게도 NAACL Findings 2025에 제1저자로 논문을 발표했고, Hugging Face Transformers에도 여러 차례 기여할 기회를 얻었다.

최근에는 다음과 같은 활동을 꾸준히 이어오고 있다.

* EleutherAI 커뮤니티 연구 프로젝트 참여
* Stanford CS336 스터디 참여 및 강의 노트 정리
* Hugging Face Transformers 오픈소스 기여
* 논문 리뷰 및 연구 노트 작성
* 개인 연구 프로젝트 수행

돌아보면 꽤 다양한 활동을 하고 있지만, 결국 모두 하나의 방향을 향하고 있다.

---

# 그런데 최근 하나의 고민이 생겼다

연구를 계속할수록 이상한 점 하나를 자주 느끼게 되었다.

논문은 읽을 수 있다.

Transformer도 이해할 수 있다.

Mechanistic Interpretability 연구도 시작할 수 있다.

그런데 FlashAttention을 제대로 이해하려고 하면 GPU Architecture를 공부해야 했고,

Megatron-LM을 읽으려면 Distributed Training을 알아야 했으며,

Jacobian Lens나 Sparse Autoencoder(SAE)를 이해하려면 선형대수와 최적화 이론을 다시 공부해야 했다.

CUDA를 이해하려면 컴퓨터 구조와 운영체제까지 연결되었다.

결국 새로운 연구를 할수록 오히려 **기초의 중요성**을 더 크게 느끼게 되었다.

생각해 보면 당연한 일이었다.

전자공학을 전공한 뒤 AI 연구로 넘어왔다. 덕분에 비교적 빠르게 연구를 시작할 수 있었지만, 컴퓨터공학 전공자들이 학부에서 자연스럽게 배우는 기반 지식을 체계적으로 공부하지는 못했다.

예전에는 크게 문제되지 않았다.

하지만 연구 주제가 점점 모델 내부 구조와 시스템으로 향하면서, 그 빈 공간이 점점 더 크게 보이기 시작했다.

그래서 앞으로는 **연구를 멈추고 공부하는 것**이 아니라,

**연구를 계속하면서 부족한 기반을 하나씩 채워 가는 과정**을 시작하려고 한다.

---

# 내가 가고 싶은 방향

장기적인 목표는 분명하다.

> **세계 최고 수준의 AI·NLP Research Scientist가 되는 것.**

단순히 기존 모델을 잘 활용하는 연구자가 아니라,

AI가 어떻게 동작하는지 이해하고,

새로운 모델과 학습 방법을 제안하며,

AI 연구 자체의 발전에 기여하는 연구자가 되고 싶다.

현재는 Large Language Models가 가장 큰 연구 대상이지만, 장기적으로는 Foundation Models와 차세대 AI 전반을 연구하는 것이 목표다.

앞으로 3년은 그 기반을 만드는 시간이 될 것이라고 생각한다.

그동안 이루고 싶은 목표는 다음과 같다.

### 연구

* 국제 학회 논문 지속 발표
* 대표 연구 분야 구축
* 연구 포트폴리오 강화
* 오픈소스 및 연구 커뮤니티 활동 지속

### 학업

* 박사 과정 진학

### 연구자로서의 성장

좋은 연구는 좋은 환경과 좋은 동료들 속에서 나온다고 생각한다.

앞으로도 더 도전적인 연구를 수행할 수 있는 환경에서 성장하며, AI 연구에 의미 있는 기여를 하는 Research Scientist가 되는 것을 목표로 한다.

이를 위해 연구 역량을 꾸준히 발전시키는 것은 물론, 컴퓨터공학의 기본기(CS Fundamentals), AI Systems, 그리고 수학적 기반(Mathematics)까지 체계적으로 갖춰 나가려고 한다.

---

# Research Focus

앞으로 몇 년간 집중하고 싶은 연구 분야는 다음과 같다.

* Foundation Models
* Model Architecture
* Mechanistic Interpretability
* AI Systems

현재 진행하고 있거나 탐색하고 있는 연구 주제는 다음과 같다.

* Transformer Architecture
* Mechanistic Interpretability
* Multilingual Reasoning
* Efficient Training

관심사는 조금씩 바뀔 수 있다.

하지만 **"AI가 어떻게 동작하는지 이해하고, 더 뛰어난 AI를 만드는 연구"**라는 방향은 앞으로도 변하지 않을 것이라고 생각한다.

---

# 앞으로 공부할 것

연구를 더 잘하기 위해 앞으로 체계적으로 공부하려는 분야도 정리해 두려고 한다.

## CS Fundamentals

연구를 하면서 가장 크게 부족함을 느낀 부분이다.

앞으로는 다음 분야를 체계적으로 공부하려고 한다.

* Algorithms & Data Structures
* Computer Architecture
* Operating Systems

이것은 면접을 위한 공부가 아니다.

더 깊은 연구를 수행하기 위한 기반을 만드는 과정이다.

---

## Mathematics

AI 연구를 위해 필요한 수학도 다시 정리하려고 한다.

특히 다음 분야를 중점적으로 공부할 계획이다.

* Linear Algebra
* Probability & Statistics
* Optimization

수학을 위한 수학이 아니라, 연구를 더 깊이 이해하고 새로운 아이디어를 만들기 위한 수학을 목표로 한다.

---

## AI Systems

앞으로의 AI 연구자는 모델뿐 아니라 시스템도 이해해야 한다고 생각한다.

그래서 다음 분야도 꾸준히 공부하려고 한다.

* GPU Architecture
* CUDA
* Distributed Training
* PyTorch Internals
* Profiling & Benchmarking

---

# 원칙

앞으로도 하나의 원칙은 변하지 않을 것이다.

> **연구가 중심이고, 공부는 연구를 더 잘하기 위한 수단이다.**

공부 자체를 목표로 삼지는 않으려고 한다.

알고리즘을 공부하는 이유는 더 나은 문제 해결 능력을 갖추기 위해서이고,

컴퓨터 구조를 공부하는 이유는 AI 시스템을 깊이 이해하기 위해서이며,

수학을 공부하는 이유는 새로운 모델과 이론을 제안하기 위해서이다.

모든 공부는 결국 더 좋은 연구를 하기 위한 과정이다.

그래서 앞으로도 다음 네 가지 원칙을 지키려고 한다.

* **Research never stops.**
* **Build strong fundamentals.**
* **Think deeply, implement rigorously.**
* **Publish impactful research.**

이 글은 완성된 계획이 아니라 현재 시점의 스냅샷이다.

아마 1년 뒤에는 연구 주제도, 관심 분야도 조금씩 바뀌어 있을 것이다.

하지만 한 가지는 변하지 않았으면 좋겠다.

**AI를 더 깊이 이해하고, 더 나은 AI를 만드는 연구를 하고 싶다는 마음.**

앞으로도 이 블로그에는 논문 리뷰, 구현 기록, 연구 노트, 그리고 시행착오를 꾸준히 기록하려고 한다.

몇 년 뒤 다시 이 글을 읽었을 때, 얼마나 성장했는지 스스로 확인할 수 있기를 기대한다.