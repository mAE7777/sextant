"""Chinese human view of the same application query used by agents."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sextant.domain import CommandResult, canonical


def _brief(value: Any, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else canonical(value)
    return text if len(text) <= limit else text[:limit] + "…（完整内容见 --json）"


def chinese(result: CommandResult) -> str:
    lines = [f"Sextant · {result.mission_id or '任务列表'}", f"状态：{result.status}"]
    if result.current_revision:
        lines.append(f"当前版本：{result.current_revision}")
    if result.errors:
        lines.extend(f"未完成：{e['code']} — {e['message']}" for e in result.errors)
        if any(e['code'] == 'runtime_unavailable' for e in result.errors):
            lines.append("恢复：检查当前运行时的可执行文件设置与 PATH；用 sextant guide runtime 查看运行条件，排除已记录的原因后再运行。")
    out = result.outputs
    if out.get('format') == 'sextant-export/1':
        lines.append(f"导出包：{out['format']}；版本 {out.get('schema_version')}；{len(out.get('revisions', []))} 个对象版本；{len(out.get('events', []))} 个事件；{len(out.get('artifacts', {}))} 个 artifacts。")
        lines.append("导出仅保存数据，不执行动作或扩大授权。")
        return "\n".join(lines) + "\n"
    if 'missions' in out:
        lines.append(f"任务：{len(out['missions'])} 个")
        lines.extend(f"  {row['mission_id']} · {row['status']} · 当前版本 {row['revision']}" for row in out['missions'])
    state = out.get("state", {})
    if state:
        lines.append(f"任务运行状态：{state['status']}；历史策略：{len(state.get('history', []))}；待执行实验：{len(state.get('pending_experiments', []))}")
        if state.get("blocker"):
            blocker = state["blocker"]
            lines.append(f"受阻：{blocker['code']} — {blocker['message']}（排除原因后再运行；已保存步骤与回执保留）")
            if blocker['code'] == 'runtime_unavailable' and not result.errors:
                lines.append("恢复：检查当前运行时的可执行文件设置与 PATH；运行 sextant guide runtime。")
        for change in state.get('history', [])[-3:]:
            if change.get('reason'):
                lines.append(f"变化原因：{_brief(change['reason'])}；策略内容变化：{canonical(change.get('changed'))}")
        for queued in state.get('pending_experiments', []):
            experiment = queued.get('experiment', {})
            lines.append(f"待执行实验 [{queued.get('branch', '未指定分支')}] {experiment.get('key', '')}：{_brief(experiment.get('purpose', '未记录目的'))}")
            lines.append(f"  执行端：{experiment.get('adapter', '未记录')}；动作：{experiment.get('action', '未记录')}；输入：{_brief(experiment.get('inputs', {}))}")
            lines.append(f"  区分：{_brief(experiment.get('distinguishes', []))}；预期观察：{_brief(experiment.get('expected_observations', []))}；改变世界：{canonical(experiment.get('changes_world', False))}")
    for item in out.get("objects", []):
        revision = item["revision"]
        kind = revision["ref"]["object_kind"]
        current = item["current_validity"]["valid"]
        payload = revision["payload"]
        if kind == "mission":
            lines.extend([f"目标：{_brief(payload['goal'])}", f"目标来源：{payload['goal_source']}"])
            lines.extend(f"硬约束：{_brief(v)}" for v in payload.get("hard_constraints", []))
        elif kind == 'proposal':
            if state.get('proposal_ref') != revision['ref']:
                continue
            label = '当前提案，适用范围见证据与核验记录' if current else '已失效，仅供追溯'
            lines.append(f"当前理解（{label}）：{_brief(payload['understanding'])}")
            lines.extend(f"关键未知：{_brief(v)}" for v in payload.get('unknowns', []))
            for candidate in payload.get('candidates', []):
                lines.append(f"备选机制（候选建议）[{candidate['key']}]：{_brief(candidate['mechanism'])}")
                lines.extend(f"  候选假设：{_brief(v)}" for v in candidate.get('assumptions', []))
                if candidate.get('temporary_disadvantage'):
                    lines.append(f"  暂时劣势：{_brief(candidate['temporary_disadvantage'])}")
                if candidate.get('combination_dependencies'):
                    lines.append(f"  组合依赖：{_brief(candidate['combination_dependencies'])}")
        elif kind == 'context' and state.get('context_ref') == revision['ref']:
            lines.extend(f"替代解释：{_brief(v)}" for v in payload.get('alternative_explanations', []))
            lines.extend(f"上下文缺口：{_brief(v)}" for v in payload.get('omissions', []))
            lines.extend(f"上下文限制：{_brief(v)}" for v in payload.get('loss', []))
        elif kind == "policy":
            if state.get("policy_ref") != revision["ref"]:
                continue  # previous proposals remain available in JSON/history
            lines.append(f"策略（{'当前依赖有效' if current else '已失效，仅供追溯'}）：{_brief(payload['summary'])}")
            lines.extend(f"条件行动：{_brief(b)}" for b in payload["branches"])
            lines.extend(f"策略假设：{_brief(v)}" for v in payload.get('assumptions', []))
            lines.extend(f"未决：{_brief(v)}" for v in payload["unresolved"])
            lines.extend(f"反转条件：{_brief(v)}" for v in payload["reversal_triggers"])
            lines.append(f"排序依据：{payload['ordering']}；{_brief(payload['basis'])}")
            if not state.get('pending_experiments'):
                lines.extend(f"策略提议的实验（排队或执行情况见任务状态/回执）：{_brief(v)}" for v in payload.get('next_experiments', []))
        elif kind == "observation":
            content = canonical(payload["content"])
            if len(content) > 1200:
                content = content[:1200] + "…（完整内容见 inspect --json）"
            lines.append(f"观察 {revision['ref']['object_id']} [{payload['provenance']}/{payload['status']}] {payload['source']}：{content}")
            lines.append(f"支持范围：{payload['supports_scope']}")
            lines.append(f"来源完整性：{payload.get('completeness', 'unknown')}；测量协议：{payload.get('measurement_protocol', 'unknown')}")
        elif kind == "prediction":
            lines.append(f"预测 [{payload['status']}]：{payload['question']}；期限：{payload['horizon']}；结算：{payload['resolver']}")
            if payload['status'] == 'resolved':
                lines.append('  结果：' + _brief(payload.get('outcome')))
            else:
                lines.append(f"  结果：尚未结算为事实（{payload['status']}）；结果域：{_brief(payload.get('outcome_domain', []))}")
            if payload.get('resolution_refs'):
                lines.append(f"  结算证据：{_brief(payload['resolution_refs'])}")
            invalidation = item.get('contract_invalidation')
            if invalidation is not None:
                lines.append(f"  冻结规范无效：{_brief(invalidation['validator_message'])}；未获得测量，不是结果失败或新增确认样本。")
                lines.append(f"  留痕原因：{_brief(invalidation['reason'])}；原版本：{_brief(invalidation['old_ref'])}")
            correction = payload.get('resolution_correction')
            if correction is not None:
                lines.append(f"  保存的事后更正：{correction['disposition']}；新增确认样本标记：{canonical(correction['confirming_sample'])}")
                lines.append(f"  原结果：{_brief(correction['original_outcome'])}；原版本：{_brief(correction['supersedes_prediction_ref'])}")
                lines.append(f"  更正原因：{_brief(correction['correction_reason'])}；原冻结时间：{correction['original_frozen_at']}")
                if correction['disposition'] == 'invalid-contract':
                    lines.append(f"  原结算规范不可用，旧结果仅供追溯：{_brief(correction['contract_error'])}")
        elif kind == 'judgment':
            lines.append(f"模型判断 [{payload.get('card_id', '未记录角色')}]：{payload.get('authority', 'unknown')}，建议；不能单独核验事实、扩大授权或批准效果。")
        elif kind == 'certificate' and payload.get('authority') == 'deterministic':
            label = '确定性核验（当前有效）' if current else '历史确定性记录（已失效，不提供当前核验）'
            lines.append(f"{label}：{_brief(payload.get('result'))}")
            lines.append(f"  适用域：{_brief(payload.get('domain', '未记录'))}；仅核验该确定性结果，不授予外部动作权限。")
        elif kind == "authorization":
            lines.append(f"执行授权：{payload['status']}；可用工具：{', '.join(payload['tools']) or '无'}")
        if not current and kind in ("policy", "proposal", "observation", "prediction", "certificate"):
            lines.extend(f"失效原因：{r}" for r in item["current_validity"]["reasons"])
    receipts: list[dict[str, Any]] = out.get("saved_receipts", [])
    if receipts:
        lines.append(f"已保存执行回执：{len(receipts)}（回执与当前结论有效性分别核验）")
        counts = Counter((row["executor"], row["state"]) for row in receipts)
        lines.extend(f"  {executor}：{status} × {count}" for (executor, status), count in sorted(counts.items()))
        for row in receipts:
            receipt = row.get("receipt") or {}
            if row["state"] != "completed":
                lines.append(f"  未完成 {row['attempt_id']}：{'; '.join(receipt.get('errors', [])) or '等待结果/对账'}")
    if result.next_actions:
        lines.append("下一步：" + " / ".join(result.next_actions))
    for key in ('invalidated', 'history_preserved', 'effects_executed', 'resolution', 'failed_operation', 'completed_steps_preserved'):
        if key in out:
            lines.append(f"{key}：{_brief(out[key])}")
    if out.get('contract_invalidation'):
        lines.append(f"冻结规范无效：{_brief(out['contract_invalidation']['validator_message'])}；未获得测量，不是新增确认样本。")
    if out and not any(key in out for key in ('state', 'objects', 'missions')):
        lines.append('其他结果字段：' + ' / '.join(out) + '；完整数据见 --json。')
    return "\n".join(lines) + "\n"
