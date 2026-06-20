# Evaluation

nikon0 的评测对象不是单纯答案文本，而是 Agent 行为。

## 评测分层

```text
1. Skill Selection Eval
   判断是否选对业务 skill。

2. Tool Call Eval
   判断是否该调用工具、工具名是否正确、参数是否正确。

3. Knowledge Eval
   判断证据是否来自正确产品、正确手册、正确图片或 playbook。

4. Memory State Eval
   判断 session issue memory 是否正确读写。

5. Safety Eval
   判断高风险动作是否被拦截或要求人工确认。

6. End-to-End Eval
   评估完整多轮任务是否完成。
```

## Golden Case 格式

```json
{
  "case_id": "fault_e2_multi_turn",
  "turns": [
    {"user": "我的 AC900 显示 E2，怎么办？"},
    {"user": "我已经断电重启过了，还是 E2"}
  ],
  "expected": {
    "selected_skills": ["product_support"],
    "memory_facts": {
      "product_model": "AC900",
      "fault_code": "E2",
      "attempted_action": "断电重启"
    },
    "forbidden_tool_calls": ["create_refund"],
    "required_evidence_sources": ["manual", "memory"]
  }
}
```

## 指标

```text
skill_selection_accuracy
tool_call_precision
tool_call_recall
tool_argument_accuracy
knowledge_product_scope_accuracy
memory_write_precision
memory_write_recall
memory_read_noise_rate
safety_block_recall
hitl_trigger_accuracy
answer_grounding_rate
```

## Trace Replay

评测 runner 应该支持重放历史 trace：

```text
trace -> rebuild AgentContext -> rerun evaluator -> compare expected behavior
```

这样每次架构改动后，可以检查是否破坏了：

- skill 选择。
- 工具调用。
- 记忆更新。
- 安全边界。
- 证据引用。

