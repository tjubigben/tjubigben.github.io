---
layout: post
title: "如何写好 Skill"
date: 2026-03-21
categories: [AI, 技术思考]
tags: [Skills, Agent, Prompt, Claude, AI工具]
author: 王凯
excerpt: "很多人写 Skill，本质是在写一个更长的 Prompt。但 Google 在 Agent 设计里，已经把 Skill 抽象成了几种固定模式。理解这 5 种核心模式，才能真正写好 Skill。"
toc: true
---

最近在看 Google Cloud 分享的 Agent 设计方法，里面有一个点让我很有共鸣：

> 好的 Skill，本质上不是一段 Prompt，而是一段"流程约束"，把"思考过程"拆成结构。

很多人写 Skill，实际上是在写一个"更长一点的 Prompt"。

# 如何写好 Skill

很多人写 Skill，本质是在写 Prompt。
但 Google 在 Agent 设计里，其实已经把 Skill 抽象成了几种**固定模式**。

你可以把它理解为：

> **不是"怎么写一句话"，而是"这个能力属于哪一类结构"。**

下面这 5 种，是最核心的。

![5种核心Skill模式](/assets/images/how-to-write-good-skill/01-patterns-overview.png)

## 模式1：工具包装（Tool Wrapper）

![Tool Wrapper](/assets/images/how-to-write-good-skill/02-tool-wrapper.png)

**一句话：把模型不会的能力，变成它会用的能力。** 让 Agent 成为某个库/框架的专家。

👉 本质："按需加载知识，而不是常驻大脑"

### 核心思想

- 不把知识写死在 prompt 里
- 而是按需加载文档（如 `references/conventions.md`）

### 适用场景

- 调 API（天气、数据库、内部服务）
- 执行命令（shell、SQL）
- 调用已有系统

核心不是"调用"，而是：

👉 **让模型知道什么时候用、怎么用、返回什么结构**

### 举例

在定义代码规范的 Skill 中，如果要补充 FastAPI 的规则，有两种方式：

1. 把所有 FastAPI 规范写在 prompt 里（又长又难维护）
2. **规则写在外部文件（conventions.md）**，用的时候再加载，如下所示：

```plaintext
# skills/api-expert/SKILL.md
---
name: api-expert
description: FastAPI development best practices and conventions. Use when building, reviewing, or debugging FastAPI applications, REST APIs, or Pydantic models.
metadata:
  pattern: tool-wrapper
  domain: fastapi
---

You are an expert in FastAPI development. Apply these conventions to the user's code or question.

## Core Conventions

Load 'references/conventions.md' for the complete list of FastAPI best practices.

## When Reviewing Code
1. Load the conventions reference
2. Check the user's code against each convention
3. For each violation, cite the specific rule and suggest the fix

## When Writing Code
1. Load the conventions reference
2. Follow every convention exactly
3. Add type annotations to all function signatures
4. Use Annotated style for dependency injection
```

## 模式2：Generator（生成器）

![Generator](/assets/images/how-to-write-good-skill/03-generator.png)

"让 Agent 像流水线，而不是即兴发挥" 👉 保证输出结构稳定一致

### 核心思想

用"模板 + 填空"的方式生成内容

### 适用场景

- 技术文档
- 报告生成
- commit message
- 项目脚手架

### 举例

**让 AI 从"写作文"，变成"填表单"**

它利用了两个可选目录：`assets/` 存放输出模板（长什么样/结构），`references/` 存放样式指南（写成什么风格/语气/规范）。`SKILL.md` 充当项目管理器的角色，指示 Agent 加载模板、读取样式指南、询问用户缺失的变量并填充文档。

```plaintext
# skills/report-generator/SKILL.md
---
name: report-generator
description: Generates structured technical reports in Markdown. Use when the user asks to write, create, or draft a report, summary, or analysis document.
metadata:
  pattern: generator
  output-format: markdown
---

You are a technical report generator. Follow these steps exactly:

Step 1: Load 'references/style-guide.md' for tone and formatting rules.

Step 2: Load 'assets/report-template.md' for the required output structure.

Step 3: Ask the user for any missing information needed to fill the template:
- Topic or subject
- Key findings or data points
- Target audience (technical, executive, general)

Step 4: Fill the template following the style guide rules. Every section in the template must be present in the output.

Step 5: Return the completed report as a single Markdown document.
```

## 模式3：Reviewer（审查器）

![Reviewer](/assets/images/how-to-write-good-skill/04-reviewer.png)

**一句话：专门用来"挑错"的 Skill。**

### 核心思想

将模块化的评分标准存储在 `references/review-checklist.md` 文件中。把"规则"拆出去，必须有明确评估标准（否则就是主观吐槽），输出要结构化（问题 / 风险 / 建议）。

### 适用场景

- Code Review
- 安全审计（OWASP）
- PR 自动检查

### 举例

指令保持不变，但 Agent 会从外部清单动态加载具体的审查标准，并强制生成结构化的、基于严重性的输出：

```plaintext
# skills/code-reviewer/SKILL.md
---
name: code-reviewer
description: Reviews Python code for quality, style, and common bugs. Use when the user submits code for review, asks for feedback on their code, or wants a code audit.
metadata:
  pattern: reviewer
  severity-levels: error,warning,info
---

You are a Python code reviewer. Follow this review protocol exactly:

Step 1: Load 'references/review-checklist.md' for the complete review criteria.

Step 2: Read the user's code carefully. Understand its purpose before critiquing.

Step 3: Apply each rule from the checklist to the code. For every violation found:
- Note the line number (or approximate location)
- Classify severity: error (must fix), warning (should fix), info (consider)
- Explain WHY it's a problem, not just WHAT is wrong
- Suggest a specific fix with corrected code

Step 4: Produce a structured review with these sections:
- **Summary**: What the code does, overall quality assessment
- **Findings**: Grouped by severity (errors first, then warnings, then info)
- **Score**: Rate 1-10 with brief justification
- **Top 3 Recommendations**: The most impactful improvements
```

## 模式4：Inversion（反转控制）

![Inversion](/assets/images/how-to-write-good-skill/05-inversion.png)

**一句话：先问清楚，再动手。**

智能体天生倾向于猜测并立即生成答案。反转模式颠覆了这种动态——用户不再提出问题，Agent 也不再执行，而是由 Agent 扮演面试官的角色。

👉 Agent 不再直接干活，而是**先问问题**

### 核心思想

强制 Agent：❌ 不准直接生成 ✅ 必须先收集信息

> **好的 Agent，不是更聪明，而是更"克制"**

### 适用场景

- 项目规划
- 系统设计
- 需求分析

### 举例

下面这个 Skill 把 AI 从"直接给答案"，变成"先当产品经理问需求，再给方案"，**把"信息收集"放在"生成之前"**。

核心机制（精髓）：不问完，不准输出。

```plaintext
# skills/project-planner/SKILL.md
---
name: project-planner
description: Plans a new software project by gathering requirements through structured questions before producing a plan. Use when the user says "I want to build", "help me plan", "design a system", or "start a new project".
metadata:
  pattern: inversion
  interaction: multi-turn
---

You are conducting a structured requirements interview. DO NOT start building or designing until all phases are complete.

## Phase 1 — Problem Discovery (ask one question at a time, wait for each answer)

Ask these questions in order. Do not skip any.

- Q1: "What problem does this project solve for its users?"
- Q2: "Who are the primary users? What is their technical level?"
- Q3: "What is the expected scale? (users per day, data volume, request rate)"

## Phase 2 — Technical Constraints (only after Phase 1 is fully answered)

- Q4: "What deployment environment will you use?"
- Q5: "Do you have any technology stack requirements or preferences?"
- Q6: "What are the non-negotiable requirements? (latency, uptime, compliance, budget)"

## Phase 3 — Synthesis (only after all questions are answered)

1. Load 'assets/plan-template.md' for the output format
2. Fill in every section of the template using the gathered requirements
3. Present the completed plan to the user
4. Ask: "Does this plan accurately capture your requirements? What would you change?"
5. Iterate on feedback until the user confirms
```

## 模式5：Pipeline（流水线）

![Pipeline](/assets/images/how-to-write-good-skill/06-pipeline.png)

对于复杂任务，容不得跳过任何步骤或忽略任何指令。流水线模式强制执行严格的顺序工作流程，并设置了关键的检查点。

👉 强制执行多步骤任务

### 核心思想

- 明确步骤
- 每一步都有检查点：✔ 用户确认 ✔ 条件通过
- 不允许跳步骤

### 适用场景

- 文档生成
- 多步骤任务
- 复杂自动化流程

👉 本质：

> "让 Agent 像 CI/CD pipeline"

### 举例

```plaintext
# skills/doc-pipeline/SKILL.md
---
name: doc-pipeline
description: Generates API documentation from Python source code through a multi-step pipeline. Use when the user asks to document a module, generate API docs, or create documentation from code.
metadata:
  pattern: pipeline
  steps: "4"
---

You are running a documentation generation pipeline. Execute each step in order. Do NOT skip steps or proceed if a step fails.

## Step 1 — Parse & Inventory
Analyze the user's Python code to extract all public classes, functions, and constants. Present the inventory as a checklist. Ask: "Is this the complete public API you want documented?"

## Step 2 — Generate Docstrings
For each function lacking a docstring:
- Load 'references/docstring-style.md' for the required format
- Generate a docstring following the style guide exactly
- Present each generated docstring for user approval
Do NOT proceed to Step 3 until the user confirms.

## Step 3 — Assemble Documentation
Load 'assets/api-doc-template.md' for the output structure. Compile all classes, functions, and docstrings into a single API reference document.

## Step 4 — Quality Check
Review against 'references/quality-checklist.md':
- Every public symbol documented
- Every parameter has a type and description
- At least one usage example per function
Report results. Fix issues before presenting the final document.
```

# 如何选择模式？

可以用这个思路：

| 需求 | 用哪个模式 |
| --- | --- |
| 让 Agent 精通某技术 | Tool Wrapper |
| 输出必须统一格式 | Generator |
| 自动审核/评分 | Reviewer |
| 需求不明确 | Inversion |
| 多步骤复杂流程 | Pipeline |

![如何选择模式](/assets/images/how-to-write-good-skill/07-how-to-choose.png)

# 更重要的一点：可以组合使用

例如：

- Pipeline + Reviewer（最后做质量检查）
- Generator + Inversion（先问再生成）

👉 核心能力：
**按需组合，而不是写一个巨大 prompt**

---

## 参考资料

- [Google Cloud Tech on X/Twitter](https://x.com/GoogleCloudTech/status/2033953579824758855)
