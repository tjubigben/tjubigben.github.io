---
layout: post
title: "Claude Code 真的会用吗"
date: 2026-03-25
categories: [ai, tools]
tags: [claude, ai-tools, development]
author: 王凯
toc: true
math: false
---

![Claude Code Hero](/assets/images/claude-code-usage-guide/claude_code_hero.png)

## 一、CLAUDE.MD：最被低估的功能

如果你只记住这篇文章的一件事，就记这个：**写好你的 CLAUDE.md**。

CLAUDE.md 是 Claude Code 在每次对话开始时自动读取的指令文件。它就像你给一个新同事写的 onboarding doc——你希望他知道什么，你就写什么。

很多人不写 CLAUDE.md，或者随便写两行。结果就是每次对话都要从头解释项目结构、编码规范、技术栈选择。这就像每天早上都要重新给同事介绍一遍公司。

### 一个好的 CLAUDE.md 应该包含什么

```markdown
# 项目名称

## 构建和测试命令

- 安装依赖：`npm install`
- 运行测试：`npm test`
- 单个测试：`npm test -- --grep "test name"`
- 格式化：`npm run format`

## 编码规范

- Python 使用 ruff 格式化，行宽 88
- 测试用 pytest，每个 service 对应一个测试文件
- API 路由文件名用复数：users.py, orders.py
- 提交信息用英文，格式：`type(scope): description`

## 架构决策

- 选 Tailwind 而不是 CSS Modules，因为团队统一了这个规范
- 用户权限校验在 middleware 里做，不要在每个路由重复写
- Redis 缓存的 key 前缀统一用 `app:v1:`

## 常见陷阱

- 数据库连接池上限是 20，别在循环里开新连接
- 不要 mock 数据库，上次 mock 测试通过但生产迁移失败了
```

### 有几个关键原则

**第一，写 Claude 从代码里读不出来的东西。** 项目的"为什么"比"是什么"更重要。你不需要解释 React 怎么用，但你需要告诉它"我们选 Tailwind 是因为团队统一了这个规范"。

**第二，控制在 200 行以内。** 官方文档明确提到，CLAUDE.md 太长会导致 Claude 忽略规则。用 markdown 标题和列表，保持可扫描性。

**第三，不要放频繁变化的内容。** 详细的 API 文档、逐文件描述这些东西不适合放在 CLAUDE.md 里。放链接就好。

### CLAUDE.md 的四级层级

![CLAUDE.md 四级层级](/assets/images/claude-code-usage-guide/claude_config_pyramid.png)

Claude Code 支持四级 CLAUDE.md，按优先级从高到低：

1. **当前目录的 CLAUDE.md**
2. **项目根目录的 CLAUDE.md**
3. **用户主目录的 ~/.claude/CLAUDE.md**
4. **系统默认配置**

我的全局 CLAUDE.md 里通常会写：

- 我是一名全栈工程师，不需要过度解释基础概念
- 回复尽量简洁，不要加无关的客套话
- 代码改动后不要总结你做了什么，我会看 diff
- 优先用简单方案，不要过度工程

这四行，能省掉你几百次重复纠正的时间。

### 进阶：用 .claude/rules/ 组织规则

当项目大了之后，一个 CLAUDE.md 文件塞不下所有规则。官方提供了一个更优雅的方案：把规则拆到 `.claude/rules/` 目录下。

```
.claude/rules/
├── testing.md      # 测试规范
├── api-style.md    # API 编写风格
└── frontend.md     # 前端约定
```

更强大的是，你可以用 YAML frontmatter 把规则限定到特定文件：

```yaml
---
paths: ["src/api/**/*.ts", "src/routes/**/*.ts"]
---

API 路由必须包含输入验证和错误处理。
所有新端点需要在 tests/api/ 下添加集成测试。
```

这样这条规则只在 Claude 访问匹配的文件时才会加载，节省上下文空间。

---

## 二、子 Agent：Claude Code 的分身术

![子 Agent 架构](/assets/images/claude-code-usage-guide/diagram3.png)

子 Agent 是 Claude Code 的一个强大机制——它可以启动独立的 AI 进程来并行处理任务，每个子 Agent 有自己的上下文窗口，不会污染你的主对话。

### 内置的子 Agent 类型

| 类型 | 能力 | 用途 |
|------|------|------|
| Explore | 只读，快速搜索代码库 | 探索、找文件、找定义 |
| Plan | 只读，研究分析 | Plan Mode 下的代码分析 |
| General | 完整能力 | 复杂的多步骤任务 |

### 子 Agent 最大的价值：隔离上下文

当你让 Claude 跑测试、分析日志、搜索大量文件时，这些操作会产生海量输出，塞满你的上下文窗口。用子 Agent 来做这些事，输出留在子 Agent 的上下文里，只有摘要返回给你。

比如：

- "用子 Agent 跑一下全量测试，把失败的用例列出来"
- "用子 Agent 搜索所有使用了 deprecated API 的文件"

### 后台运行：Ctrl+B

按 `Ctrl+B` 可以把子 Agent 放到后台运行。你可以继续和 Claude 聊其他事，等后台任务完成后会自动通知你。

**适合：**

- 跑测试套件（通常需要几分钟）
- 大范围代码搜索
- 不紧急的代码审查

### 自定义子 Agent

你可以在 `.claude/agents/` 目录下创建自定义 Agent：

```yaml
name: code-reviewer
description: 代码审查专家。代码改动后自动触发。
tools: Read, Grep, Glob, Bash
model: sonnet
```

你是一位资深代码审查者。检查以下维度：

- 代码清晰度和可读性
- 错误处理是否完整
- 有没有暴露敏感信息
- 输入验证
- 性能隐患

然后在对话中用 `@"code-reviewer (agent)"` 调用它。

---

## 三、上下文管理：最容易被忽视的关键

![上下文管理](/assets/images/claude-code-usage-guide/diagram4.png)

Claude Code 的上下文窗口虽然大（最多 1M token），但它不是无限的，而且上下文管理直接决定了输出质量。

### 上下文里都装了什么

- 你的对话历史
- Claude 读取的所有文件内容
- 命令执行的输出
- CLAUDE.md 文件（每次都加载）
- Memory 文件（前 200 行）
- 加载的 Skill 和 MCP 工具定义

### 五个实用策略

1. **`/clear`：一件事做完就清**

   不要在同一个对话里又修 bug 又加功能又重构。`/clear` 会清空上下文但保留 CLAUDE.md，相当于一次免费的重启。

2. **`/compact`：手动压缩上下文**

   当对话变长时，输入 `/compact` 让 Claude 自动总结和压缩之前的对话。更好的用法是给压缩加一个焦点：

   ```
   /compact 专注于 API 改动和测试结果
   ```

   这样 Claude 在压缩时会优先保留你关心的内容。

3. **`/context`：看看上下文被谁吃了**

   输入 `/context` 可以看到当前上下文的使用情况——哪些文件占了多少 token，MCP 工具定义占了多少。我经常发现一些没用的 MCP server 占了大量空间。

4. **用子 Agent 隔离噪声**

   跑测试、分析日志这些操作会产生大量输出。让子 Agent 去做，主对话的上下文保持干净。

5. **在 CLAUDE.md 里写压缩保留指令**

   你可以告诉 Claude 压缩时必须保留什么：

   ```markdown
   # 压缩指令

   压缩上下文时，始终保留：

   - 已修改文件的完整列表
   - 测试命令和结果
   - 关键的架构决策
   ```

### Memory vs CLAUDE.md

**Memory 适合存什么：** 你的偏好（"这个人喜欢简洁回复"）、项目约定（"部署有个特殊步骤"）、历史决策（"上次选了方案 A 是因为 X"）。

**Memory 不适合存什么：** 代码细节、Git 历史、临时状态——这些从代码和 Git 里获取更准确。

用 `/memory` 命令可以查看和管理所有已加载的 Memory。

---

## 四、Extended Thinking：让 Claude 想深一点

对于复杂问题，你可以开启 Extended Thinking 模式，让 Claude 在回答前花更多时间推理。

### 怎么用

```bash
# 快捷键切换
Option+T (Mac) / Alt+T (Windows/Linux)

# 或者用命令
/effort high   # 更深入的推理
/effort max    # 最大推理深度
/effort low    # 简单任务，省 token
```

### 什么时候该用

- 复杂的架构决策
- 棘手的 debug（多个可能原因需要排除）
- 多步骤的重构方案设计
- 权衡多个方案的利弊

### 什么时候不需要

- 简单的代码修改
- 格式化、重命名
- 已经很明确的 bug 修复

**小技巧：** 在提示词里写 `ultrathink` 可以强制触发最高推理深度，不需要手动调整 effort 设置。

---

## 五、Hooks：让规则变成铁律

![Hooks 机制](/assets/images/claude-code-usage-guide/diagram5.png)

CLAUDE.md 里的指令是"建议"——Claude 大部分时候会遵守，但偶尔会忘。Hooks 是"铁律"——无论如何都会执行。

Hooks 是在特定生命周期事件上自动触发的 shell 命令。配置在 `settings.json` 里。

### 关键事件类型

| 事件               | 说明       |
| ---------------- | -------- |
| PreToolUse       | 工具执行前    |
| PostToolUse      | 工具执行后    |
| SessionStart     | 会话开始时    |
| UserPromptSubmit | 用户提交提示词时 |

### 实用示例

**自动格式化**——每次编辑后运行 Prettier：

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
      }]
    }]
  }
}
```

**保护关键文件**——阻止修改生产配置：

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": ".claude/hooks/protect-files.sh"
      }]
    }]
  }
}
```

Hook 命令返回 exit code 0 表示允许，exit code 2 表示阻止。这意味着你可以写任意复杂的判断逻辑。

**上下文压缩后重新注入关键信息：**

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "compact",
      "hooks": [{
        "type": "command",
        "command": "echo '用 Bun 不用 npm。提交前跑 bun test。'"
      }]
    }]
  }
}
```

这解决了一个常见痛点：对话太长被压缩后，之前提过的重要指令可能丢失。用 Hook 可以在每次压缩后自动重新注入。

### Hook 的四种类型

| 类型 | 说明 |
|------|------|
| command | 执行 shell 命令，从 stdin 读取 JSON |
| http | 向 URL 发送 POST 请求 |
| prompt | 单次 LLM 调用，做 yes/no 判断 |
| agent | 启动一个子 Agent 进行验证 |

---

## 六、Git 工作流：用好 Worktree

![Git Worktree](/assets/images/claude-code-usage-guide/diagram6.png)

Claude Code 能执行 Git 命令，这是一把双刃剑。但官方提供了一个很好的安全网：Worktree。

### Git Worktree：隔离的工作空间

```bash
# 在隔离的 worktree 中启动 Claude
claude --worktree feature-auth
claude --worktree bugfix-123

# 自动生成名字
claude --worktree
```

Worktree 会在 `.claude/worktrees/` 下创建一个独立的 Git 分支副本。Claude 在里面怎么折腾都不影响你的主分支。退出时：

- 如果没有改动 → 自动清理
- 如果有改动 → 提示你保留或删除

这对于探索性任务特别有用。不确定某个重构方案能不能行？开个 worktree 试试，不行就扔掉，零风险。

### 几个 Git 铁律

1. **永远不要让 Claude Code 自动 push**

   它可以 commit，但 push 这个动作应该由你来确认。一旦 push 到远端，回退成本就大了。

2. **频繁 commit**

   做完一个小功能就 commit。用 `/rewind` 可以回退到任意一个 checkpoint，但前提是你有 commit 记录。

3. **警惕破坏性操作**

   如果 Claude Code 要执行 `git reset --hard`、`git push --force`、`rm -rf`，一定要在脑子里过一遍后果再批准。这些操作没有 undo。你也可以在权限规则里直接 deny 掉这些命令。

4. **从 PR 恢复上下文**

   ```bash
   claude --from-pr 123
   ```

   这会自动加载 PR 的改动和讨论作为上下文，非常适合 code review 或者继续别人的工作。

---

## 七、审查代码：信任但要验证

Claude Code 写的代码质量总体不错，但你不能盲目信任。

![代码审查](/assets/images/claude-code-usage-guide/diagram7.png)

### 几个常见问题

1. **过度工程**

   你让它写一个简单的工具函数，它给你搞出一个完整的抽象层，带泛型、带接口、带工厂模式。杀鸡用了牛刀。

   **解决方法：** 在 CLAUDE.md 里写上"优先用简单方案"，或在指令里明确说"不需要抽象"。

2. **幻觉 API**

   它有时候会用不存在的 API 或者过时的语法。尤其是小众库的新版本。

   **解决方法：** 跑测试。这也是为什么指令里应该包含验证标准。

3. **改动范围膨胀**

   你让它改一个函数，它把整个文件都重构了。

   **解决方法：** 明确说"只改 X，不要动其他代码"。或者用 `/freeze` 命令限制编辑范围到指定目录。

### Writer/Reviewer 模式

官方推荐的一个高级模式：用两个独立的会话分别扮演"写代码"和"审查代码"的角色。

两个会话有独立的上下文，Reviewer 不受 Writer 的思路影响，能发现 Writer 的盲点。

- **会话 A（Writer）：**"实现 API 限流中间件"
- **会话 B（Reviewer）：**"审查 @src/middleware/rateLimiter.ts 里的限流实现，重点看边界情况、竞态条件、一致性"
- **会话 A：**"修复审查反馈：【粘贴 B 的输出】"

---

## 八、不要用 Claude Code 做的事

说了这么多"应该怎么做"，最后聊聊"不应该做什么"。

1. **不要让它做你完全不了解的事**

   如果你对 Kubernetes 一无所知，不要让 Claude Code 帮你写部署配置然后直接用。你无法审查你不懂的东西。

2. **不要在没有版本控制的环境下用**

   没有 Git 就没有回退的能力。Claude Code 的改动可能覆盖你的文件，没有版本控制就是裸奔。

3. **不要一口气扔一个巨大的任务**

   "把整个项目从 JavaScript 迁移到 TypeScript" 这种指令等于让 Claude Code 自由发挥。结果不可控。拆成小步骤，每一步确认后再做下一步。

4. **不要指望一次就完美**

   迭代是正常的。用 `/rewind` 回退，用精确的反馈描述"哪里不对"。

---

## 总结

Claude Code 的核心使用哲学其实很简单：

> 它是一个极其能干的协作者，但不是自动驾驶。方向盘始终在你手里。

按重要性排序，我的建议是：

1. **写好 CLAUDE.md** — 一次投入，每次对话都受益
2. **给精确的指令** — 目标 + 约束 + 验证标准
3. **用 Plan Mode** — 复杂任务先规划再动手
4. **管理上下文** — `/clear`、`/compact`、子 Agent
5. **控制权限** — deny 危险操作，allow 常用命令
6. **频繁 commit** — 保留回退能力

做到这些，Claude Code 能让你的效率翻好几倍。做不到，它只会更快地帮你制造混乱。

**工具的价值，永远取决于使用它的人。**

---

## 附录 A：实战案例

![实战案例](/assets/images/claude-code-usage-guide/diagram8.png)

### 案例 1：复杂重构的安全流程

**背景：** 需要将一个 2000 行的单体服务拆分为 3 个独立模块。

**策略：**
1. 开启 Plan Mode 让 Claude 先出方案
2. 用 Worktree 创建隔离环境
3. 分阶段执行，每阶段完成后验证

**具体指令：**
```bash
# 第一步：创建隔离环境
claude --worktree refactor-service

# 第二步：在 Plan Mode 下分析
/plan
"分析这个服务的依赖关系，给出拆分为 auth、order、payment 三个模块的方案，
要求：
- 保持现有 API 兼容
- 每个模块独立可测试
- 给出迁移步骤和回滚方案"

# 第三步：按方案逐步执行
# 每完成一个模块就运行测试验证
```

**结果：** 3 小时完成重构，零回归 bug。

---

### 案例 2：用 Extended Thinking 解决并发 Bug

**背景：** 生产环境偶发的竞态条件，日志无法定位。

**策略：**
1. 开启 Extended Thinking（Option+T）
2. 提供完整的上下文：代码 + 日志 + 系统架构

**具体指令：**
```
/effort high

"这个订单状态更新存在竞态条件：
1. 代码：【粘贴相关代码】
2. 日志显示两个请求同时修改同一订单
3. 数据库隔离级别是 READ COMMITTED

分析可能的竞争场景，给出修复方案。
要求考虑：
- 乐观锁 vs 悲观锁的选择
- 对性能的影响
- 是否需要分布式锁"
```

**结果：** Claude 识别出"检查-更新"模式的竞态，建议使用数据库行级锁 + 重试机制。

---

### 案例 3：子 Agent 批量处理

**背景：** 需要给 50 个 API 路由统一添加输入验证。

**策略：**
用子 Agent 并行处理，主对话保持清爽。

**具体指令：**
```
"启动子 Agent 完成以下任务：
1. 找出 src/routes/ 下所有缺少输入验证的 .ts 文件
2. 对每个文件添加对应的 zod schema 验证
3. 返回修改的文件列表和每个文件的改动摘要

约束：
- 不要修改已有的验证逻辑
- 保持原有代码风格
- 每个文件改动后运行类型检查"
```

**结果：** 15 分钟完成全部修改，主对话只收到 10 行摘要。

---

## 附录 B：FAQ 与排错

### CLAUDE.md 相关问题

**Q：为什么 Claude 有时不遵守 CLAUDE.md？**

可能原因：
1. 文件超过 200 行，被 Claude 忽略部分规则
2. 使用了模糊的表述，如"尽量""可能"
3. 规则之间存在冲突

**解决方法：**
- 用 `/context` 检查 CLAUDE.md 是否被加载
- 将关键规则写成祈使句："总是...""不要..."
- 用 Hooks 强制执行关键规则

---

**Q：上下文压缩后丢失了关键信息怎么办？**

在 CLAUDE.md 中添加压缩指令：
```markdown
# 压缩指令

压缩上下文时，始终保留：
- 已修改文件的完整列表
- 测试命令和结果
- 关键的架构决策
```

或使用 Hook 在压缩后重新注入：
```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "compact",
      "hooks": [{
        "type": "command",
        "command": "echo '关键规则：...'"
      }]
    }]
  }
}
```

---

### 子 Agent 相关问题

**Q：子 Agent 和主对话的边界如何把握？**

建议分工：

| 任务类型 | 使用 | 原因 |
|---------|------|------|
| 探索代码库 | 子 Agent | 输出量大，避免污染主上下文 |
| 运行测试 | 子 Agent | 可能产生大量日志 |
| 核心逻辑修改 | 主对话 | 需要你的直接确认 |
| 代码审查 | 子 Agent | 独立视角，发现盲点 |

---

**Q：子 Agent 执行结果不符合预期？**

检查清单：
1. 指令是否足够具体？（目标 + 约束 + 验证标准）
2. 是否提供了足够的上下文？
3. 是否使用了 `@` 引用相关文件？

---

### Hooks 相关问题

**Q：Hooks 配置不生效？**

排查步骤：
1. 检查 `settings.json` 位置：`~/.claude/settings.json`
2. 验证 JSON 格式是否正确
3. 检查命令是否有执行权限
4. 查看 Claude Code 的日志输出

调试技巧：先用简单的 `echo` 测试 Hook 是否触发：
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit",
      "hooks": [{
        "type": "command",
        "command": "echo 'Hook triggered' >> /tmp/claude-hook.log"
      }]
    }]
  }
}
```

---

### 常见陷阱

**陷阱 1：过度信任**
- 现象：直接让 Claude 执行 `git push` 或数据库迁移
- 后果：生产事故
- 解决：始终在本地验证后再手动 push

**陷阱 2：上下文爆炸**
- 现象：对话越来越慢，响应质量下降
- 原因：积累了太多无关的文件和输出
- 解决：定期 `/clear` 或使用子 Agent

**陷阱 3：指令模糊**
- 现象：Claude 反复修改，始终不符合预期
- 原因：指令缺少约束条件
- 解决：明确"做什么" + "不做什么" + "验收标准"

---

## 附录 C：进阶技巧

### 1. MCP 工具整合

Claude Code 支持 Model Context Protocol (MCP)，可以接入自定义工具。

配置示例（`~/.claude/mcp.json`）：
```json
{
  "mcpServers": {
    "database": {
      "command": "node",
      "args": ["/path/to/db-mcp-server.js"],
      "env": {
        "DB_URL": "postgresql://..."
      }
    }
  }
}
```

使用场景：
- 直接查询数据库 schema
- 调用内部 API 获取上下文
- 集成公司内部的文档系统

---

### 2. 与 Git Hooks 配合

在 `.claude/hooks/` 中创建脚本，与项目 Git Hooks 联动：

```bash
#!/bin/bash
# .claude/hooks/pre-commit-check.sh

# 阻止提交包含 TODO 的代码
if grep -r "TODO" src/ --include="*.ts"; then
  echo "Error: 发现 TODO，请先处理"
  exit 2
fi

exit 0
```

---

### 3. 快速启动模板

为不同类型的任务创建别名：

```bash
# ~/.zshrc
alias claude-refactor='claude --worktree refactor-$(date +%s)'
alias claude-review='claude --from-pr $(git rev-parse --abbrev-ref HEAD)'
alias claude-explore='claude --no-git'
```

---

## 附录 D：我的配置模板

### CLAUDE.md 模板

```markdown
# 项目规范

## 沟通偏好
- 回复简洁，不要解释显而易见的代码
- 代码改动后不要总结，我会看 diff
- 遇到不确定的问题先询问，不要假设

## 技术栈
- 语言：TypeScript / Node.js
- 框架：Express
- 数据库：PostgreSQL + Prisma
- 测试：Vitest

## 编码规范
- 使用严格的 TypeScript 配置
- 所有函数必须有返回类型注解
- 错误处理统一使用自定义 AppError 类
- 数据库查询必须用参数化查询

## 架构决策
- 业务逻辑放在 service 层，不在 controller 写逻辑
- 所有外部调用（DB、API）必须包装在 repository 层
- 不要过早抽象，先实现再重构

## 测试要求
- 新功能必须有单元测试
- 集成测试放在 tests/integration/
- 测试覆盖率不低于 80%

## 禁止事项
- 不要修改 .env 文件
- 不要执行数据库删除操作
- 不要自动 push 代码
```

### settings.json 模板

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "npm run lint:fix"
      }]
    }],
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": ".claude/hooks/protect-main-branch.sh"
      }]
    }]
  },
  "permissions": {
    "allow": [
      "npm test",
      "npm run build",
      "git status",
      "git diff"
    ],
    "deny": [
      "git push",
      "git reset --hard",
      "rm -rf",
      "drop database",
      "delete from"
    ]
  }
}
```

---

## 写在最后

Claude Code 不是银弹。它不能替代你的判断，不能替你承担责任。

但它确实是一个极其高效的**放大器**——把你的意图快速转化为代码，把你的思路迅速验证为结果。

关键在于：**你要清楚自己要什么，然后精确地告诉它。**

希望这篇文章能帮你少走弯路，多写代码。
