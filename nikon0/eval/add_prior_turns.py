"""为 10 条 multi-turn 用例添加 prior_turns，将伪多轮转为真实 session 回放数据."""

from __future__ import annotations

import json
from pathlib import Path

DATASET_PATH = Path("nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl")

# 每条 multi-turn item 的 turn 1（prior_turn）设计
# 当前 message 自然成为 turn 2
PRIOR_TURNS: dict[str, dict] = {
    "qa-126": {
        "prior_turns": ["我的空气炸锅有问题，不加热了"],
        # turn2 = "我的空气炸锅有问题"（当前 message）
        # 更新 golden：agent 在 turn1 后应已询问具体故障，turn2 用户仍未说清 → agent 继续追问
        "golden_answer": "请具体描述一下空气炸锅的故障现象：是完全不加热、加热不均匀、显示屏异常还是有其他表现？不同的故障原因不同，我需要了解具体症状才能帮您排查。",
        "answer_must_contain": ["问题", "故障", "具体"],
    },
    "qa-127": {
        "prior_turns": ["我家空调完全不制冷了，需要报修。地址是浦东新区张江路100号，电话13800138000"],
        # turn2 = "上次报修的空调，维修师傅来了说修好了，但用了两天又不制冷了"
        "golden_answer": "很抱歉维修后再次出现同样问题。这说明之前的维修可能未解决根本原因。我已调出您之前的维修记录。我们将安排高级技师重新上门检修，更换的配件和人工不另收费（属于维修保修范围）。同时我们将把此案例标记为重复维修跟进。",
        "answer_must_contain": ["再次故障", "维修", "不制冷"],
    },
    "qa-128": {
        "prior_turns": ["My microwave smells bad after cooking. How do I clean it?"],
        # turn2 = "I followed your cleaning instructions for the microwave but the smell still hasn't gone away..."
        "golden_answer": "If the smell persists after cleaning, check the charcoal filter. If your microwave is vented inside, the charcoal filter should be replaced every 6-12 months and cannot be cleaned. Also check the grease filter (clean at least monthly). Run the vent fan on HI for 30 minutes after cooking. Place a bowl of water with lemon juice in the microwave and heat for 3 minutes — the steam helps neutralize odors.",
        "answer_must_contain": ["odor", "charcoal filter", "vent"],
    },
    "qa-129": {
        "prior_turns": ["我家洗碗机底部漏水，是什么问题？"],
        # turn2 = "上次你说洗碗机漏水是门封条的问题，我换了门封条但还是漏，而且漏得更多了"
        "golden_answer": "更换门封条后漏水加重，可能原因：1) 门封条安装不到位或型号不匹配，2) 漏水源头并非门封条而是其他部件（如进水阀、排水管连接处、内胆密封），3) 门铰链松动导致关门后密封不严。建议停止使用并安排专业技术人员上门检测确认真正漏水源。请勿再自行维修以免影响保修。",
        "answer_must_contain": ["门封条", "更换", "仍然漏水"],
    },
    "qa-130": {
        "prior_turns": ["蓝牙鼠标连不上电脑了，怎么办？"],
        # turn2 = "按照你说的换了新电池也重新配对了，蓝牙鼠标还是连不上"
        "golden_answer": "已更换电池且重新配对仍无法连接时，请尝试以下深度排查：1) 到控制面板>鼠标>蓝牙查看鼠标电量状态，2) 卸载并重新安装WIDCOMM蓝牙驱动程序，3) 确认电脑蓝牙功能未被禁用，4) 使用USB蓝牙接收器（而非电脑自带蓝牙）尝试配对。若以上均无效，鼠标可能存在硬件故障，可联系经销商获取技术支持或申请保修。",
        "answer_must_contain": ["新电池", "配对", "连不上"],
    },
    "qa-131": {
        "prior_turns": ["Nespresso咖啡机出来的咖啡有苦涩味，怎么处理？"],
        # turn2 = "咖啡机除垢后还是有苦涩味，而且出咖啡速度变慢了"
        "golden_answer": "除垢后仍有苦涩味和出咖啡变慢：1) 确保已用清水充分冲洗机器（可能需要多冲洗一个循环），2) 苦涩味可能来自咖啡胶囊本身而非机器，请检查胶囊保质期，3) 出咖啡变慢可能出口堵塞，用软湿布清洁咖啡出口，4) 检查水箱是否使用了新鲜饮用水（非存放多日的水）。如问题持续请联系Nespresso Club。",
        "answer_must_contain": ["苦涩味", "速度慢", "清洁"],
    },
    "qa-132": {
        "prior_turns": ["我家空调制冷效果很差，不够冷怎么办？"],
        # turn2 = "上次你教我清洁了空调滤网，制冷好了一点但还是不够冷，还有别的原因吗？"
        "golden_answer": "滤网清洁后制冷仍不足可能是以下原因：1) 制冷剂不足或泄漏（需专业人员检测），2) 室外机散热不良（周围有遮挡物或冷凝器脏污），3) 房间面积超过空调制冷能力，4) 温度传感器故障导致压缩机未正常工作。建议安排技术人员上门检测制冷剂压力和系统运行参数。",
        "answer_must_contain": ["滤网", "不够冷", "其他原因"],
    },
    "qa-133": {
        "prior_turns": ["我家冰箱温度忽冷忽热不稳定，是什么问题？"],
        # turn2 = "冰箱温度还是不稳定，上次让我检查的门封条我检查了没问题"
        "golden_answer": "门封条正常但温度仍不稳定时，建议排查：1) 冰箱是否频繁开门或门未关严，2) 是否放入大量未冷却的热食导致箱内温度升高，3) 冷凝器盘管是否积灰影响散热，4) 温控器或温度传感器是否异常，5) 冰箱周围通风空间是否足够（背面和侧面需要散热空间）。如均正常仍需安排技术人员上门检测制冷系统。",
        "answer_must_contain": ["温度不稳定", "门封条", "继续排查"],
    },
    "qa-134": {
        "prior_turns": ["我用PS VR头显玩游戏一会就头晕，怎么办？"],
        # turn2 = "之前关于VR头晕你说适应一下就好了，我已经试了两周了还是不行。还有其他办法吗？"
        "golden_answer": "适应两周后仍持续出现VR眩晕，建议尝试：1) 从短时间（5-10分钟）开始，逐步延长，2) 优先选择低运动强度的VR内容（如360度视频而非第一人称动作游戏），3) 确保游玩区域光线充足、通风良好，4) 调整瞳距设置使镜片与眼睛距离合适，5) 尝试使用晕车药（请先咨询医生）。如果所有方法都无效，此产品可能不适合您的体质，可考虑联系购买渠道了解退货选项。",
        "answer_must_contain": ["头晕", "两周", "适应"],
    },
    "qa-135": {
        "prior_turns": ["225B吹风机怠速时老是熄火，怎么调？"],
        # turn2 = "吹风机按你说的清洗了空气滤清器也调了化油器，现在怠速不熄火了但是高转速没力"
        "golden_answer": "怠速改善但高转速无力，调节方法：1) 全开油门，调节H油针直至发动机达到最大转速，2) 将H油针逆时针旋1/8圈（若限位限制则少于1/8圈）。注意：调节H油针时全油门持续时间最长10秒，之后需怠速至少10秒。另外检查燃油混合是否正确（1:50），机油过多会导致动力下降和火花塞积碳。",
        "answer_must_contain": ["化油器", "高转速", "无力"],
    },
}


def apply_prior_turns(dataset_path: Path, output_path: Path | None = None) -> Path:
    output_path = output_path or dataset_path

    lines = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cid = item["case_id"]

            if cid in PRIOR_TURNS:
                update = PRIOR_TURNS[cid]
                item["turns"] = "multi"
                item["prior_turns"] = update["prior_turns"]
                item["golden_answer"] = update.get("golden_answer", item.get("golden_answer", ""))
                item["expected"]["answer_must_contain"] = update.get(
                    "answer_must_contain",
                    item["expected"].get("answer_must_contain", []),
                )

            lines.append(json.dumps(item, ensure_ascii=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def verify(dataset_path: Path) -> None:
    """打印更新后的 multi-turn 项供检查."""
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["category"] != "multi-turn":
                continue
            pt = item.get("prior_turns", [])
            print(f"{item['case_id']}:")
            print(f"  turns={item.get('turns', '?')}, prior_count={len(pt)}")
            for i, msg in enumerate(pt, 1):
                print(f"  T{i}: {msg}")
            print(f"  T{len(pt)+1}: {item['message'][:80]}")
            print(f"  must_contain: {item['expected'].get('answer_must_contain', [])}")
            print()


if __name__ == "__main__":
    apply_prior_turns(DATASET_PATH)
    verify(DATASET_PATH)
