# Case Intake MCP

把客服项目里的 `CaseIntakeSkill` 暴露为 MCP 服务，供 `mcp-gateway` 统一转发。

## Run

```bash
cd /Users/nikonzhang/compeletion
conda run -n kefu python examples/mcp_case_intake/server.py
```

默认 endpoint：

```text
http://127.0.0.1:8011/mcp
```

Docker 内的 `java-mcp-gateway` 访问宿主机时使用：

```text
http://host.docker.internal:8011/mcp
```

## Tools

- `collect_case_intake`: 收集售后/报修/退款工单槽位。
- `cancel_case_intake`: 取消指定 session 的未完成工单草稿。

