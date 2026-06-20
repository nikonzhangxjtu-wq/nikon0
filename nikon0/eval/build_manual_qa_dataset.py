"""构建 150 条基于真实手册内容的 QA eval 题目。

类别分布:
  product_support: 30
  troubleshooting:  25
  case_intake:      20
  refund:           10
  handoff:          10
  composite:        15
  boundary:         15
  multi-turn:       10
  general:          10
  no_evidence:       5
"""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path("/Users/nikonzhang/compeletion/nikon0/eval/datasets/agent_qa_eval_150_manual.jsonl")
MANUAL_DIR = "手册"

ITEMS = []

def add(case_id, category, message, must_contain, golden, manual, product="", turns="single", acceptable_skills=None, language="zh", handoff=False, needs_approval=False):
    ITEMS.append({
        "case_id": case_id,
        "category": category,
        "message": message,
        "turns": turns,
        "expected": {
            "acceptable_skills": acceptable_skills or ["product_support"],
            "answer_must_contain": must_contain,
            "handoff": handoff,
            "needs_approval": needs_approval,
        },
        "golden_answer": golden,
        "metadata": {
            "source_manual": f"{MANUAL_DIR}/{manual}",
            "product_name": product,
            "language": language,
        },
    })

# ============ product_support (30 items) ============

# Airfryer
add("qa-001", "product_support",
    "空气炸锅怎么连接WiFi和手机App？",
    ["2.4 GHz", "NutriU", "WiFi"],
    "将空气炸锅插入电源，确保智能设备在家庭WiFi范围内。下载NutriU App，在'My Appliances'中选择连接的空气炸锅，按照App指示完成WiFi连接和配对。WiFi LED常亮表示连接成功。需连接2.4 GHz 802.11 b/g/n家庭WiFi。",
    "Airfryer.txt", "Airfryer")

add("qa-002", "product_support",
    "How do I make homemade fries in the Airfryer?",
    ["potato", "soak", "800g", "oil"],
    "Peel potatoes and cut into 10x10mm sticks. Soak in water for at least 30 minutes. Dry with towel. Mix with one tablespoon of cooking oil. Put sticks in basket and fry, shaking 2-3 times during cooking. Best to air fry in portions up to 800g.",
    "Airfryer.txt", "Airfryer", language="en")

add("qa-003", "product_support",
    "空气炸锅的保温模式怎么用？能保温多久？",
    ["保温", "30分钟", "keep warm"],
    "按菜单按钮直到保温图标闪烁，按电源按钮启动保温模式。保温定时器默认设置为30分钟，可通过时间下调按钮更改保温时间（1-30分钟）。保温模式下温度不可更改。",
    "Airfryer.txt", "Airfryer")

add("qa-004", "product_support",
    "空气炸锅使用后如何清洁？",
    ["清洁", "洗碗机", "不粘涂层", "冷却"],
    "清洁前让炸篮、炸锅及设备内部完全冷却。炸锅和炸篮可在洗碗机中清洗，也可用热水、洗洁精和非研磨海绵清洁。不要使用金属厨具或研磨性清洁材料以免损坏不粘涂层。每次使用后应清洁并清除锅底油脂。",
    "Airfryer.txt", "Airfryer")

add("qa-005", "product_support",
    "What are the safety warnings for the Airfryer?",
    ["do not", "oil", "steam", "hot"],
    "Do not fill the pan with oil. Do not cover air inlet and outlet openings. Keep hands and face at safe distance from steam. Accessible surfaces may become hot. Never immerse in water. Unplug immediately if dark smoke appears. Do not touch hot surfaces. Always place on dry, stable, level surface.",
    "Airfryer.txt", "Airfryer", language="en")

# 吹风机
add("qa-006", "product_support",
    "225B吹风机使用什么燃油？混合比是多少？",
    ["汽油", "二冲程机油", "1:50", "辛烷值"],
    "本设备配备二冲程发动机，必须使用汽油与二冲程机油的混合燃油。推荐使用胡斯华纳二冲程机油，混合比为1:50。使用辛烷值不低于87的优质汽油。严禁使用水冷船外机专用二冲程机油或四冲程发动机专用机油。",
    "吹风机手册.txt", "225B吹风机")

add("qa-007", "product_support",
    "吹风机启动前需要做哪些检查？",
    ["停机开关", "油门锁", "减震系统"],
    "操作前需检查吹风机状态，重点检查消音器、进气口及空气滤清器。检查停机开关功能是否正常。检查油门锁与油门扳机安全运行状态。确保所有防护装置在位。检查所有螺母与螺钉是否紧固。检查所有机壳是否无裂纹。",
    "吹风机手册.txt", "225B吹风机")

add("qa-008", "product_support",
    "吹风机使用时有哪些安全要求？",
    ["10米", "儿童", "废气", "防护装备"],
    "禁止无关人员及动物进入作业区域（距操作人员10米范围内）。禁止未成年人操作。操作人员需佩戴听力防护、眼部防护装备、防滑工作靴。严禁将吹风机喷口对准人或动物。严禁在通风不良的空间操作。加油前必须关闭发动机。站在梯子或支架上时严禁操作。",
    "吹风机手册.txt", "225B吹风机")

add("qa-009", "product_support",
    "吹风机的化油器怎么调节？",
    ["化油器", "低速油针", "高速油针", "怠速调节", "3000"],
    "化油器设有三处调节装置：H=高速油针、L=低速油针、T=怠速调节螺钉。顺时针旋入油针使空燃比变稀，逆时针旋出变浓。推荐怠速转速为3000转/分钟。调节时全油门持续时间最长10秒，之后需怠速至少10秒。H油针逆时针旋1/8圈。",
    "吹风机手册.txt", "225B吹风机")

# 冰箱
add("qa-010", "product_support",
    "冰箱温度应该设置多少度合适？",
    ["冷藏室", "冷冻室", "温度", "调节"],
    "冷藏室建议设置在3-5°C，冷冻室建议设置在-18°C至-20°C。夏季可适当调低温度，冬季可适当调高。温度调节需根据环境温度和食物储存量进行调整。",
    "冰箱手册.txt", "冰箱")

add("qa-011", "product_support",
    "冰箱第一次使用需要注意什么？",
    ["静置", "清洁", "通电", "空载"],
    "新冰箱搬运后需静置2-4小时再通电。首次使用前需清洁冰箱内部。通电后先空载运行2-3小时，待箱内温度降低后再放入食物。热食需冷却至室温后再放入冰箱。",
    "冰箱手册.txt", "冰箱")

# 空调
add("qa-012", "product_support",
    "空调的定时功能怎么设置？",
    ["定时", "开机", "关机", "遥控器"],
    "通过遥控器上的定时按钮设置。可以设置定时开机或定时关机，时间范围通常为0.5-24小时。设置后空调将在指定时间自动开启或关闭。",
    "空调手册.txt", "空调")

add("qa-013", "product_support",
    "空调自诊断功能怎么用？",
    ["自诊断", "故障代码", "指示灯"],
    "空调具有自诊断功能。当出现故障时，室内机或室外机的指示灯会按照特定规律闪烁。通过指示灯闪烁次数可以判断故障代码。具体故障代码需查阅手册中的故障代码表。",
    "空调手册.txt", "空调")

add("qa-014", "product_support",
    "空调滤网多久清洁一次？怎么清洁？",
    ["滤网", "清洁", "两周", "吸尘器"],
    "建议每两周清洁一次空调滤网。清洁时可使用吸尘器吸除灰尘，或用清水冲洗后晾干再装回。滤网脏污会影响制冷/制热效果并增加能耗。",
    "空调手册.txt", "空调")

# 洗碗机
add("qa-015", "product_support",
    "洗碗机用什么盐？怎么添加？",
    ["专用盐", "软化器", "盐仓"],
    "洗碗机需使用专用的洗碗机盐，不可使用食用盐。盐用于软化水质。添加时将盐倒入底部的专用盐仓，首次使用时需先加水。盐仓一般位于洗碗机底部滤网下方。",
    "洗碗机手册.txt", "洗碗机")

add("qa-016", "product_support",
    "洗碗机的亮碟剂和洗碗粉分别放哪里？",
    ["亮碟剂", "洗碗粉", "分配器"],
    "洗碗粉放入门内侧的洗涤剂分配器中，亮碟剂（漂洗剂）加入专用的亮碟剂分配器。亮碟剂有助于加速干燥和防止水渍。分配器有刻度显示剩余量。",
    "洗碗机手册.txt", "洗碗机")

# 烤箱
add("qa-017", "product_support",
    "烤箱如何进行自清洁？",
    ["自清洁", "高温", "200°C", "热风循环"],
    "烤箱空载，开启热风循环功能，设定200°C运行约一小时。随后待设备冷却，用海绵清除食物残留。催化侧面板带有特殊微孔搪瓷涂层可吸附油脂飞溅物。请勿使用腐蚀性或研磨性清洁剂以免损坏催化表面。",
    "烤箱手册.txt", "烤箱")

add("qa-018", "product_support",
    "烤箱的旋转烤叉怎么用？",
    ["旋转烤叉", "家禽", "鸡肉", "接油盘"],
    "用于均匀烤制大块肉类及家禽。将肉类穿在烤叉上，鸡肉可用棉线捆扎，固定牢固后插入烤箱前壁对应座内。建议在第一层放置加有半升水的接油盘收集肉汁。烤叉的塑料手柄在烹饪前必须取下。",
    "烤箱手册.txt", "烤箱")

add("qa-019", "product_support",
    "How do I descale my Nespresso coffee machine?",
    ["descaling", "0.5 L", "water tank", "rinse"],
    "Remove capsule and close lever. Fill water tank with 0.5L drinkable water and add 1 descaling liquid. Place container (min 1L) under outlet. Press both Espresso and Lungo buttons for 3 seconds to enter descaling mode. Press Lungo button and wait until tank is empty. Refill with used solution and repeat. Then refill with fresh water and rinse. Press both buttons for 3 seconds to exit.",
    "Coffee_Machine.txt", "Nespresso", language="en")

add("qa-020", "product_support",
    "How do I program the water volume on my Nespresso machine?",
    ["water volume", "Espresso", "Lungo", "program"],
    "Turn machine on and wait for ready mode (steady lights). Fill water tank and insert a capsule. Place cup under outlet. Press and hold Espresso or Lungo button until desired volume is served. Release button. Water volume is now stored. Factory settings: Espresso 40ml, Lungo 110ml.",
    "Coffee_Machine.txt", "Nespresso", language="en")

# Microwave
add("qa-021", "product_support",
    "How do I use the sensor cook function on my microwave?",
    ["sensor", "humidity", "SENSING", "covered"],
    "Food should be at normal storage temperature. Turntable and outside of container should be dry. Foods must be covered loosely with microwaveable plastic wrap, waxed paper or a lid. Do not open door during sensing. Display shows SENSING during initial period. Oven beeps twice when sensing is done and displays remaining cook time.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

add("qa-022", "product_support",
    "How do I set the child lock on the microwave?",
    ["child lock", "STOP/CLEAR", "four seconds", "LOCKED"],
    "Touch STOP/CLEAR once. Then touch and hold 0 pad for more than four seconds. LOCKED will appear in display and you hear two beeps. To cancel, touch and hold 0 more than four seconds until LOCKED disappears.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

# Washing Machine
add("qa-023", "product_support",
    "洗衣机有哪些安全注意事项？",
    ["接地", "电源", "儿童", "金属管"],
    "必须确保洗衣机接地。使用金属管接地，但不能用燃气管或电话线接地以防爆炸或雷击。脱水时不要将手伸入脱水桶。禁止儿童操作或玩耍洗衣机。使用完毕后需断开电源线。",
    "Washing_Machine.txt", "Washing Machine")

add("qa-024", "product_support",
    "洗衣机的洗涤程序怎么操作？",
    ["洗涤选择旋钮", "水位", "洗涤定时器", "排水"],
    "设置洗涤选择旋钮至所需模式。设置水位选择旋钮至WASH。加水入洗衣桶并加入洗涤剂。放入衣物并加水至H高水位线。设置洗涤定时器1-15分钟。洗涤完成后将循环选择旋钮设至DRAIN排水。",
    "Washing_Machine.txt", "Washing Machine")

# 牙刷
add("qa-025", "product_support",
    "电动牙刷的SenseIQ功能是什么？",
    ["SenseIQ", "压力", "刷头", "力度"],
    "SenseIQ技术可感知刷牙压力、动作和覆盖范围，实时反馈以改善刷牙习惯。压力传感器在刷牙力度过大时会提醒用户减轻力度，保护牙龈。",
    "Electric_Toothbrush.txt", "电动牙刷")

# 蓝牙鼠标
add("qa-026", "product_support",
    "蓝牙激光鼠标怎么安装电池？",
    ["AA", "电池", "正负极", "电池仓"],
    "按下按钮弹出电池仓盖。按照鼠标内部标注的正负极（+和-）装入两节AA电池。装回电池仓盖。LED指示灯变为琥珀色时表示电量低，需立即更换电池。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标")

add("qa-027", "product_support",
    "蓝牙鼠标无法工作时怎么办？",
    ["电池", "BIOS", "配对", "LED"],
    "检查电池是否安装正确，正负极需与标识一致。LED指示灯红色表示电量低需更换新电池。检查系统BIOS中的USB鼠标功能是否启用。确保在干净、平整、不光滑的表面使用鼠标。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标")

# 空气净化器
add("qa-028", "product_support",
    "空气净化器的滤网多久更换一次？",
    ["滤网", "更换", "指示灯", "清洁"],
    "根据使用环境和使用频率，前置滤网可2-4周清洁一次，HEPA滤网一般6-12个月更换一次。当滤网更换指示灯亮起时需及时更换。具体更换周期取决于空气质量和使用时间。",
    "空气净化器手册.txt", "空气净化器")

add("qa-029", "product_support",
    "空气净化器的静音模式怎么开启？",
    ["静音模式", "睡眠", "低噪音", "夜间"],
    "通过模式按钮切换至睡眠/静音模式。在静音模式下，设备以最低风速运行，噪音降到最低，适合夜间使用。同时显示屏亮度会降低。",
    "空气净化器手册.txt", "空气净化器")

# VR头显
add("qa-030", "product_support",
    "VR头显使用时有哪些安全警告？",
    ["12岁", "休息", "游玩区域", "障碍物"],
    "VR头显不适用于12岁以下儿童。佩戴VR头显会限制对周围环境的视野。使用前需清除游玩区域内的所有障碍物。建议每游玩1小时休息15分钟。若出现头晕、恶心、视觉异常等症状应立即停止使用。尽可能保持坐姿。",
    "VR头显手册.txt", "VR头显")

# ============ troubleshooting (25 items) ============

add("qa-031", "troubleshooting",
    "空气炸锅冒黑烟了怎么办？",
    ["拔下电源", "黑烟", "停止"],
    "看到设备冒出黑烟时，立即拔下设备电源。等待烟雾排放停止后再将炸锅从设备中拉出。不要在冒烟时打开炸锅取出食物。",
    "Airfryer.txt", "Airfryer")

add("qa-032", "troubleshooting",
    "空气炸锅显示屏显示---是什么意思？",
    ["---", "固件更新", "stand-by"],
    "显示'---'表示空气炸锅正在进行自动固件更新。更新在待机模式下自动启动，大约需要1分钟，期间空气炸锅不能使用。更新完成后恢复正常。更新对保障隐私和设备正常运行非常重要。",
    "Airfryer.txt", "Airfryer")

add("qa-033", "troubleshooting",
    "Why won't my microwave start?",
    ["door", "Start", "power", "fuse"],
    "Check if the door is completely closed. Press START after entering cooking time. Check if power cord is plugged in and circuit breaker/fuse is working. If oven still doesn't start, check if child lock is activated.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

add("qa-034", "troubleshooting",
    "Microwave has sparking or arcing inside. What should I do?",
    ["metal", "foil", "arcing", "turntable"],
    "Remove any metal utensils, aluminum foil, or metal-trimmed dishes. Metal causes arcing which can damage the oven. If using foil, keep at least 1 inch from oven walls. Check that turntable is clean and properly seated. If arcing persists with no metal present, contact service.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

add("qa-035", "troubleshooting",
    "吹风机发动机启动困难怎么办？",
    ["火花塞", "空气滤清器", "化油器", "清洁"],
    "首先检查火花塞是否脏污，清洁并检查电极间隙（标准0.5mm）。检查空气滤清器是否需要清洁，每25小时清洁一次。检查化油器调节是否正确。若火花塞电极严重积碳，可能原因是化油器调节不当或燃油混合错误。",
    "吹风机手册.txt", "225B吹风机")

add("qa-036", "troubleshooting",
    "吹风机发动机过热怎么办？",
    ["冷却系统", "气缸散热片", "空气滤清器", "消音器"],
    "检查冷却系统是否脏污或堵塞，包括启动装置进气口、飞轮风扇叶片、气缸散热片。每周至少用刷子清洁一次。检查消音器火花阻隔网是否堵塞。滤网堵塞会导致发动机过热，损坏气缸与活塞。",
    "吹风机手册.txt", "225B吹风机")

add("qa-037", "troubleshooting",
    "Nespresso machine is not brewing coffee. What should I check?",
    ["water tank", "capsule", "lever", "power"],
    "Check water tank is filled with fresh drinking water. Ensure capsule is properly inserted and lever is fully closed. Check machine is turned on (steady lights). If lights are blinking, machine is heating up (takes about 25 seconds). Check power connection. If capsule is stuck, unplug and call Nespresso Club.",
    "Coffee_Machine.txt", "Nespresso", language="en")

add("qa-038", "troubleshooting",
    "洗衣机不排水了怎么办？",
    ["排水", "排水过滤器", "异物", "硬币"],
    "检查排水过滤器是否堵塞。拆下排水过滤器，清除滤网中的棉绒和异物。检查排水管是否弯折或被压。确认排水管安装高度正确（带泵型号约70-80cm）。",
    "Washing_Machine.txt", "Washing Machine")

add("qa-039", "troubleshooting",
    "洗碗机不启动是什么原因？",
    ["电源", "门", "进水", "程序"],
    "检查电源是否接通，门是否完全关闭锁紧。检查水龙头是否打开，进水软管是否弯折。确认程序已正确选择并按下启动键。若以上正常仍不启动，可能为控制板或门锁故障。",
    "洗碗机手册.txt", "洗碗机")

add("qa-040", "troubleshooting",
    "空调不制冷了怎么办？",
    ["滤网", "制冷剂", "温度设定", "室外机"],
    "首先检查空调滤网是否脏污堵塞需要清洁。确认温度设定是否低于室温，模式是否在制冷档。检查室外机是否正常运行，周围是否有障碍物影响散热。如果以上都正常，可能是制冷剂泄漏需联系售后。",
    "空调手册.txt", "空调")

add("qa-041", "troubleshooting",
    "冰箱不制冷了怎么排查？",
    ["电源", "温度设置", "门封", "通风"],
    "先检查电源是否正常，冰箱灯是否亮。确认温度设置是否正确（未被误调高）。检查门封条是否密封良好。检查冰箱周围通风空间是否充足，冷凝器是否积灰。长时间开门会导致温度升高。",
    "冰箱手册.txt", "冰箱")

add("qa-042", "troubleshooting",
    "烤箱门打不开怎么办？",
    ["关闭", "自清洁", "自动解锁", "冷却"],
    "关闭烤箱后重新开机查看故障是否仍存在。如果是自清洁期间，烤箱门将锁定无法打开，需等待自动解锁。自清洁结束后待烤箱冷却门锁才会释放。若显示屏显示字母'F'加数字，请联系最近售后服务中心。",
    "烤箱手册.txt", "烤箱")

add("qa-043", "troubleshooting",
    "烤箱不工作了怎么排查？",
    ["市电", "电源", "F", "程序器"],
    "检查市电是否正常，烤箱是否已接通电源。关闭烤箱后重新开机查看故障是否仍存在。若电子程序器显示字母'F'加数字，请联系最近售后服务中心并说明'F'后的数字。",
    "烤箱手册.txt", "烤箱")

add("qa-044", "troubleshooting",
    "蓝牙鼠标连接不上电脑怎么办？",
    ["电池", "配对", "BIOS", "接收器"],
    "检查鼠标电池是否正确安装且电量充足（LED红灯表示低电量）。重新按下鼠标底部配对按钮。检查USB蓝牙接收器是否正确插入。检查系统BIOS中USB鼠标功能是否已启用。在干净平整表面使用，深色鼠标垫会增加电量消耗。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标")

add("qa-045", "troubleshooting",
    "空气净化器开机后不出风或风量很小？",
    ["滤网", "进风口", "风扇", "更换"],
    "检查进风口和出风口是否被遮挡。检查滤网是否严重脏污需要更换（滤网更换指示灯是否亮起）。确认风扇是否正常运行。如滤网使用时间过长严重堵塞会导致风量显著下降。",
    "空气净化器手册.txt", "空气净化器")

add("qa-046", "troubleshooting",
    "VR头显屏幕画面不动或显示异常怎么办？",
    ["强制关机", "电源键", "7秒", "重启"],
    "长按VR头显线控上的电源键至少7秒进行强制关机，然后重新开机。若问题依旧，重启系统。保持VR头显的附着传感器无遮挡，否则取下头显时屏幕可能不会自动关闭，导致画面残留。",
    "VR头显手册.txt", "VR头显")

add("qa-047", "troubleshooting",
    "VR头显处理器单元过热怎么处理？",
    ["通风口", "冷却", "高温", "遮挡"],
    "屏幕会显示高温提示信息。关闭系统并静置一段时间。待处理器单元冷却后，将其移至通风良好的位置再继续使用。请勿遮挡处理器单元的通风口，请勿在封闭橱柜等易积热环境中使用。",
    "VR头显手册.txt", "VR头显")

add("qa-048", "troubleshooting",
    "The microwave turntable is not rotating. What's wrong?",
    ["T/Table On/Off", "turntable", "OFF", "sensor"],
    "Check if T/Table On/Off feature has been turned off. 'OFF' will show in display when turntable is off. Note: turntable cannot be turned off during sensor cook and defrost modes. Check that roller rest is properly positioned and turntable is correctly seated.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

add("qa-049", "troubleshooting",
    "Airfryer has error code or won't turn on. What troubleshooting steps?",
    ["plug", "outlet", "switch", "reset"],
    "Check that plug is properly inserted into wall outlet. Verify outlet has power. Ensure On/Off button was pressed to switch on. If appliance was in automatic shut-off (20 minutes no button press), press On/Off button again. For factory reset, press temperature and time up buttons simultaneously for 10 seconds.",
    "Airfryer.txt", "Airfryer", language="en")

add("qa-050", "troubleshooting",
    "电动牙刷充不进电了怎么办？",
    ["充电器", "接触", "电池", "充电指示灯"],
    "检查充电器是否正确连接电源。确认牙刷正确放置在充电底座上且接触良好。检查充电指示灯是否亮起。如果长时间未使用电池完全耗尽，可能需要充电较长时间才能恢复。若以上均正常仍不充电，可能电池已损坏需维修。",
    "Electric_Toothbrush.txt", "电动牙刷")

add("qa-051", "troubleshooting",
    "洗衣机脱水时有异常噪音和振动怎么办？",
    ["平衡", "地面", "衣物", "螺钉"],
    "检查洗衣机是否放置在坚固平整的地面上（允许2°倾斜）。确保衣物在脱水桶中均匀分布。检查底脚是否正确安装且紧固。检查运输螺钉是否已拆除。过少的衣物可能导致不平衡。",
    "Washing_Machine.txt", "Washing Machine")

add("qa-052", "troubleshooting",
    "How do I clean the microwave grease filter?",
    ["grease filter", "monthly", "dishwasher", "slide"],
    "Remove grease filter by sliding to side, pulling downward and pushing to other side. Wash in dishwasher or soak in hot water with mild detergent. Rinse well and shake dry. Do not use ammonia. The aluminum filter will darken. Reinstall by sliding into side slot, pushing up and toward oven center to lock. Clean at least once a month.",
    "Microwave_OTR.txt", "Microwave OTR", language="en")

add("qa-053", "troubleshooting",
    "吹风机消音器需要怎么维护？",
    ["消音器", "火花阻隔网", "催化转化器", "钢丝刷"],
    "无催化转化器的消音器滤网每周清洁或按需更换，可用钢丝刷清洁。带催化转化器的消音器滤网每月检查清洁。滤网损坏需更换，频繁堵塞可能表明催化转化器功能异常，联系经销商检查。消音器损坏的设备严禁使用。",
    "吹风机手册.txt", "225B吹风机")

add("qa-054", "troubleshooting",
    "电子程序器显示异常怎么办？",
    ["电子程序器", "数字", "显示屏", "售后"],
    "若显示屏显示字母'F'加数字，此为故障代码。请联系最近售后服务中心，说明字母'F'后的具体数字，以便服务人员诊断问题。在联系售后服务前，可先尝试关闭设备电源后重新开机查看故障是否仍存在。",
    "烤箱手册.txt", "烤箱")

add("qa-055", "troubleshooting",
    "VR头显屏幕出现黑色暗点或常亮像素是怎么回事？",
    ["像素", "屏幕", "正常现象", "暗点"],
    "屏幕特定位置可能出现黑色（暗点）像素或常亮像素。此类斑点属于屏幕正常现象，并非系统故障。屏幕采用高精度技术制造，但每个屏幕仍可能出现极少数暗点或常亮像素。",
    "VR头显手册.txt", "VR头显")

# ============ case_intake (20 items) ============

add("qa-056", "case_intake",
    "我的AC900空调坏了不启动，需要报修，电话13800138000，地址是北京市朝阳区建国路100号",
    ["intent", "repair", "slots"],
    "已为您登记报修信息。产品型号：AC900，故障描述：空调不启动，联系电话：13800138000，地址：北京市朝阳区建国路100号。维修人员将在24小时内与您联系确认上门时间。报修单号：WO-20260619-001。",
    "空调手册.txt", "AC900空调",
    acceptable_skills=["case_intake", "product_support"])

add("qa-057", "case_intake",
    "我要投诉，买的产品质量太差了，用了三天就坏了，找你们主管来处理",
    ["投诉", "主管", "升级"],
    "非常抱歉给您带来不愉快的体验。我已记录您的投诉信息并将此案例升级至主管处理。售后服务主管将在1个工作日内与您联系。如需加快处理，您也可直接拨打客服热线。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["case_intake", "general"])

add("qa-058", "case_intake",
    "我需要把Nespresso咖啡机寄回维修，订单号ORD-98765",
    ["维修", "ORD", "return"],
    "已为您登记产品返修信息。订单号：ORD-98765，产品：Nespresso咖啡机。请将产品恢复出厂设置并清空系统后寄回。维修中心收到后将进行检测维修，预计维修周期为7-10个工作日。",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["case_intake"])

add("qa-059", "case_intake",
    "我刚搬家冰箱需要重新安装调试，能安排技术人员上门吗？",
    ["安装", "上门", "技术人员", "预约"],
    "已为您登记上门安装需求。请提供联系电话、上门地址和期望的上门时间段。冰箱搬运后需静置2-4小时再通电，技术人员会上门进行安装、调试并告知使用注意事项。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["case_intake", "product_support"])

add("qa-060", "case_intake",
    "洗碗机漏水了！水从前面门缝流出来，快安排人来修，电话13900139000，杭州市西湖区",
    ["漏水", "维修", "紧急"],
    "已为您登记紧急维修。故障现象：洗碗机门缝漏水，联系电话：13900139000，地址：杭州市西湖区。此问题可能涉及门封条损坏或进水阀故障。维修人员将优先安排，预计24小时内上门处理。请暂时停止使用以防进一步漏水。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-061", "case_intake",
    "Bluetooth mouse not working at all, need to return for replacement. Order ORD-45612, bought last week.",
    ["replacement", "ORD", "return"],
    "I've registered your replacement request. Order: ORD-45612, product: Bluetooth Laser Mouse. Please check: 1) batteries installed correctly (AA x2), 2) LED indicator status, 3) BIOS USB mouse function enabled. If confirmed defective, we'll process replacement within 3-5 business days.",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标",
    acceptable_skills=["case_intake", "troubleshooting"], language="en")

add("qa-062", "case_intake",
    "上次维修的吹风机又出问题了，同一个问题反复出现，要求换新机",
    ["重复故障", "换新", "维修记录"],
    "我理解您的情况，吹风机同一故障反复维修确实让人困扰。已查询您之前的维修记录。我将为您申请换新处理，需要您提供原始购买凭证。售后主管将审核并确认是否符合换新条件，预计2个工作日内答复。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["case_intake"])

add("qa-063", "case_intake",
    "空气炸锅的手柄摔坏了，这个能单独购买配件吗？型号是HD9200",
    ["配件", "手柄", "HD9200"],
    "空气炸锅手柄可作为单独配件购买。型号HD9200的手柄配件编号需确认具体批次。建议您联系授权售后服务中心订购原厂配件。不建议自行维修，否则保修将失效。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["case_intake", "product_support"])

add("qa-064", "case_intake",
    "VR头显的镜片被划伤了，影响使用体验，能更换镜片吗？",
    ["镜片", "划伤", "更换"],
    "VR头显镜片一般不支持单独更换，划伤严重可能需要更换整个头显单元。建议先尝试用专用软布清洁镜片确认是否为污渍而非划伤。如需进一步处理请联系授权维修中心评估损坏情况。请勿使用非镜片专用纸质产品擦拭镜片。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["case_intake", "product_support"])

add("qa-065", "case_intake",
    "烤箱买了不到一个月，烤箱灯不亮了，要保修",
    ["保修", "一个月", "烤箱灯"],
    "您的烤箱在12个月保修期内，烤箱灯不亮属于保修范围。烤箱灯泡型号为25-40W/230V E-14、耐温300°C白炽灯。您可自行更换（断开电源后操作），或联系授权服务中心安排上门更换。请提供购买凭证和服务编号（烤箱门右侧铭牌上Service字样后的编号）。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-066", "case_intake",
    "我需要取消上周三预约的上门维修服务，空调安装改成下周一",
    ["取消", "改约", "上门"],
    "已为您取消原预约的空调安装服务，并重新预约至下周一。新的预约单号已生成。如有变更请提前24小时通知我们。请确保下周一有人在家以便技术人员上门。",
    "空调手册.txt", "空调",
    acceptable_skills=["case_intake"])

add("qa-067", "case_intake",
    "我刚发现冰箱显示屏上出现了一个故障代码E5，是什么问题？需要报修吗？",
    ["E5", "故障代码", "维修"],
    "已记录您的冰箱故障咨询。故障代码E5通常表示温度传感器异常。建议您先尝试断电5分钟后重新通电看是否恢复。若故障代码持续存在，需要技术人员上门检测维修。已为您登记，可随时确认上门时间。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-068", "case_intake",
    "洗衣机电机还在保修期内但嗡嗡响不转了",
    ["保修", "电机", "不转"],
    "已为您登记保修维修。洗衣机电机不转但有嗡嗡声，可能原因为启动电容故障或电机卡死。请确认您已检查排水过滤器无异物堵塞。在保修期内，授权维修人员将免费上门维修。请提供购买日期和产品序列号。",
    "Washing_Machine.txt", "Washing Machine",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-069", "case_intake",
    "我想购买空气炸锅的炸篮替换件，型号HD9650",
    ["炸篮", "替换件", "HD9650"],
    "空气炸锅炸篮可作为原厂替换件单独购买。型号HD9650的炸篮具体配件编号请咨询授权服务中心。建议使用原厂配件以确保与产品兼容及使用安全。第三方配件可能影响保修和使用效果。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["case_intake", "product_support"])

add("qa-070", "case_intake",
    "I need to file a warranty claim for microwave that stopped heating after 3 months",
    ["warranty", "heating", "12 months", "authorized"],
    "Your microwave is within the 12-month warranty period. Since it's not heating, this could be a magnetron issue. Please provide your proof of purchase and model number (HMV9302/HMV9305/HMV9306/HMV9307). We'll arrange service through an authorized provider. Do not attempt to repair yourself as this voids the warranty.",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["case_intake", "troubleshooting"], language="en")

add("qa-071", "case_intake",
    "之前买的蓝牙鼠标，USB接收器丢了，能单独买接收器吗？",
    ["接收器", "遗失", "单独购买"],
    "蓝牙鼠标本款不含接收器单独销售。您可以选择购买新的蓝牙接收器，支持蓝牙v1.2及v2.0规范的USB蓝牙接收器均可兼容。也可以不通过接收器，使用电脑自带蓝牙功能直接与鼠标配对。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标",
    acceptable_skills=["case_intake", "product_support"])

add("qa-072", "case_intake",
    "洗碗机安装需要什么条件？我家厨房没有预留位置",
    ["安装", "进水管", "排水管", "电源"],
    "已记录您的安装需求。洗碗机安装需要：1) 靠近水槽的独立空间（标准宽度60cm），2) 冷水进水管接口，3) 排水管连接（距地面40-100cm），4) 独立接地的三孔电源插座。如无预留位置，需改造橱柜。我们可以安排安装师傅上门评估现场条件后给出方案。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["case_intake", "product_support"])

add("qa-073", "case_intake",
    "Nespresso machine leaking water from bottom, need urgent repair. Can I get a loaner machine?",
    ["leaking", "repair", "urgent"],
    "Urgent repair registered for your Nespresso machine (water leak from bottom). Please unplug immediately and do not use. This could indicate internal water circuit damage. We'll check loaner machine availability in your area. Please provide your Club membership number for priority service.",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["case_intake", "troubleshooting"], language="en")

add("qa-074", "case_intake",
    "VR头显配套的处理器单元灯不亮了，开机没反应",
    ["处理器单元", "电源", "状态指示灯"],
    "已记录您的VR硬件问题。请先确认：1) 电源适配器是否正确连接（DC IN 12V接口），2) 电源线是否完好，3) 插座是否有电。处理器单元状态指示灯应亮白色，如完全不亮可能为电源适配器故障。请使用标配电源适配器，不可使用第三方适配器。若排除电源问题仍不亮需送修。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-075", "case_intake",
    "烤箱使用时有焦味散发出来，很担心安全问题",
    ["焦味", "安全", "检查"],
    "已为您记录安全相关的报修申请。烤箱散发焦味可能原因：1) 新设备首次使用绝缘材料和防护油脂产生的正常气味（建议200°C空烤一小时），2) 底部有食物残渣或油脂在烘烤中焦化，3) 内部配件或线路异常。建议先彻底清洁烤箱内部，若焦味持续存在，请停止使用并等待技术人员上门检查。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["case_intake", "troubleshooting"])

# ============ refund (10 items) ============

add("qa-076", "refund",
    "这个空气炸锅我不想要了，买回来只用了一次，可以退货吗？",
    ["退货", "退款", "退款政策"],
    "关于退货请求，需确认是否在退货期限内（通常购买后7-15天），商品需保持原包装完整且无使用痕迹。请提供订单号，我将帮您查询是否符合退货条件，如符合将为您生成退货单号和退货地址。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["refund", "general"])

add("qa-077", "refund",
    "买的烤箱和描述不符，我要全额退款，订单号ORD-33445",
    ["退款", "ORD-33445", "全额"],
    "已记录您的退款申请。订单号ORD-33445，退款原因：商品与描述不符。此情况属于符合退款条件的场景。需要您提供商品实物照片确认问题。确认后将按原支付方式全额退款，退款到账时间取决于支付方式（通常3-7个工作日）。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["refund", "case_intake"])

add("qa-078", "refund",
    "I want a refund for this microwave. It arrived damaged with a cracked door.",
    ["refund", "damaged", "return"],
    "I'm sorry the microwave arrived damaged. This is covered under our return policy. Please keep all packaging for inspection. We'll process a full refund or arrange a replacement, whichever you prefer. Please provide your order number and photos of the damage. We'll provide a prepaid return shipping label.",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["refund", "case_intake"], language="en")

add("qa-079", "refund",
    "昨天买的蓝牙鼠标，回家发现电脑没有USB口用不了，能退吗？",
    ["退", "USB", "兼容"],
    "蓝牙鼠标可通过USB蓝牙接收器或电脑自带蓝牙连接。若您的电脑无USB接口也无自带蓝牙功能，可在退货期限内办理退货。商品需保持完好及原包装。请携带鼠标和购买凭证到原购买门店或联系在线客服办理退货。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标",
    acceptable_skills=["refund", "product_support"])

add("qa-080", "refund",
    "吹风机买错了型号，没拆封可以退换吗？",
    ["退换", "未拆封", "型号"],
    "未拆封的商品在退货期限内可以办理退换。请提供购买凭证和订单号。您可以选择换购正确型号或全额退款。换货由我们承担运费，退货需保持商品未拆封且包装完好。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["refund", "case_intake"])

add("qa-081", "refund",
    "洗碗机用了一个月发现洗不干净还有异味，要求退货",
    ["一个月", "洗不干净", "退货"],
    "您反映洗碗机使用后洗不干净且有异味。建议先检查：1) 是否使用了正确的洗碗机专用盐和亮碟剂，2) 滤网是否需要清洁，3) 喷臂是否堵塞。如排除使用问题后仍存在，超过正常退货期但仍在保修期内的产品可安排免费维修。如需进一步处理需售后人员上门检测后判定。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["refund", "troubleshooting", "case_intake"])

add("qa-082", "refund",
    "VR头显用了眼睛不舒服头晕恶心，实在用不了想退",
    ["头晕", "恶心", "退货"],
    "VR眩晕是部分用户初次使用时的常见反应。建议初次使用时短时间体验，每玩1小时休息15分钟。如果经过适应期（1-2周内每次短时间使用）后症状仍持续，可联系购买渠道咨询退货事宜。具体退货政策取决于购买时间和渠道。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["refund", "product_support"])

add("qa-083", "refund",
    "买咖啡机的时候额外买了胶囊，但胶囊发错了口味，要求退款",
    ["胶囊", "发错", "退款"],
    "很抱歉胶囊发错了口味。Nespresso胶囊仅在授权渠道购买可保证品质。发错商品我们可以为您办理换货或退款。请提供订单号和收到的胶囊口味信息，我们将安排正确的胶囊补发或退款处理，发错的胶囊您可自行保留。",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["refund", "case_intake"])

add("qa-084", "refund",
    "冰箱收到后发现门上有个凹痕，能退换吗？",
    ["凹痕", "退换", "外观"],
    "新冰箱发现外观凹痕属于运输损坏问题，符合退换货条件。建议您在收货时当场验货，如有外观问题及时拒收。如已签收，请在24小时内联系客服并提供照片，我们将安排换货或退款处理。请注意保修范围不包括外观划痕和小凹痕。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["refund", "case_intake"])

add("qa-085", "refund",
    "Could not get the Airfryer to connect to WiFi after trying for a week. Want my money back.",
    ["return", "refund", "WiFi", "connect"],
    "I understand your frustration with the WiFi connection. Before processing a return, let me confirm a few things: the Airfryer requires 2.4 GHz 802.11 b/g/n WiFi (not 5 GHz). Ensure your router's 2.4 GHz band is enabled. If you've verified this and still can't connect, we can process a return within the return period. Do you have your order number?",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["refund", "troubleshooting"], language="en")

# ============ handoff (10 items) ============

add("qa-086", "handoff",
    "我要投诉你们售后人员态度非常恶劣，工号ZH1234，不接受机器人处理，必须转人工",
    ["投诉", "人工", "转接"],
    "非常抱歉您遇到不愉快的服务体验。我已记录您的投诉（涉及员工工号ZH1234），并立即为您转接人工客服主管处理。请稍候。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["handoff", "case_intake"], handoff=True)

add("qa-087", "handoff",
    "I've been trying to fix this microwave for hours and nothing works. This is unacceptable. I want to speak to a real person now.",
    ["speak", "real person", "supervisor"],
    "I understand your frustration and will connect you to a live agent immediately. I've summarized the issue: microwave not functioning after troubleshooting attempts. A customer service supervisor will take over now. Please hold.",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["handoff", "troubleshooting"], handoff=True, language="en")

add("qa-088", "handoff",
    "你们产品漏电！我刚才用洗衣机的时候被电了一下，现在手臂还有点麻，这种情况你们必须给我一个说法！",
    ["漏电", "触电", "安全", "紧急"],
    "您的安全问题我们会最高优先级处理！请立即停止使用洗衣机并拔掉电源！我已紧急转接安全专员和售后服务主管，他们会立即与您联系。同时建议您如有身体不适请先就医。我们对此非常重视，会立即启动安全事故调查程序。",
    "Washing_Machine.txt", "Washing Machine",
    acceptable_skills=["handoff", "case_intake", "troubleshooting"], handoff=True)

add("qa-089", "handoff",
    "我已经打了三次电话了！每次都说会安排人上门修洗碗机，到现在都没人来。给我转经理！",
    ["经理", "多次", "未处理"],
    "非常抱歉您多次联系仍未得到解决。我立即为您转接客服经理，并将此案例标记为升级处理。我已整理您之前的联系记录供经理参考。经理将亲自跟进并确保维修安排落实。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["handoff", "case_intake"], handoff=True)

add("qa-090", "handoff",
    "你们的咖啡机把我厨房台面烧焦了，要求赔偿！这已经不只是维修的问题了",
    ["赔偿", "烧焦", "安全"],
    "这是一起需要紧急处理的安全和赔偿问题。我已记录：Nespresso咖啡机疑似导致厨房台面损坏。需要立即转接索赔部门和安全管理团队处理。请保持现场状态并拍照留存。理赔专员将在2小时内与您联系。请务必停止使用该咖啡机。",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["handoff", "case_intake"], handoff=True)

add("qa-091", "handoff",
    "人工客服",
    ["人工"],
    "好的，正在为您转接人工客服。请稍候。",
    "空调手册.txt", "空调",
    acceptable_skills=["handoff"], handoff=True)

add("qa-092", "handoff",
    "转人工，我不跟机器人说话，直接给我接通人工坐席",
    ["转人工", "人工坐席"],
    "正在为您转接人工坐席，请稍候。当前等待人数为1人，预计等待时间约30秒。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["handoff"], handoff=True)

add("qa-093", "handoff",
    "I need legal action. Your product caused property damage and I want to file a formal complaint with your legal department.",
    ["legal", "complaint", "formal"],
    "I understand you need to escalate this to our legal department. This is beyond my scope to handle. I will transfer you immediately to our legal and compliance team. Please have any documentation ready. A case number will be assigned for your formal complaint.",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["handoff"], handoff=True, language="en")

add("qa-094", "handoff",
    "Voglio parlare con un operatore umano, per favore. Non parlo cinese.",
    ["umano", "operatore", "italiano"],
    "Capisco che ha bisogno di un operatore umano che parli italiano. La sto trasferendo al nostro team di supporto internazionale. Un momento per favore.",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["handoff"], handoff=True, language="other")

add("qa-095", "handoff",
    "这个问题已经超出了机器人的处理范围，涉及保险理赔问题，我的洗碗机漏水把楼下邻居的天花板泡坏了",
    ["保险", "邻居", "理赔"],
    "我理解这是一个涉及第三方财产损失的复杂理赔案件。这确实超出了机器人的处理范围。我立即为您转接保险理赔专员和售后服务主管。请准备好：1) 洗碗机购买凭证，2) 漏水现场照片，3) 邻居损失情况说明。理赔专员将指导您完成整个保险理赔流程。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["handoff", "case_intake"], handoff=True)

# ============ composite (15 items) ============

add("qa-096", "composite",
    "空气炸锅E2故障代码是什么意思？另外我想查一下上次的维修记录",
    ["E2", "故障", "维修记录"],
    "关于E2故障代码，需要查阅空气炸锅手册中的故障排除部分确认具体含义。关于维修记录查询，需要您提供之前的报修单号或联系电话和姓名来查询历史记录。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-097", "composite",
    "冰箱温度显示异常，需要报修，同时帮我查一下这款冰箱的保修期还有多久",
    ["温度显示", "报修", "保修期"],
    "冰箱温度显示异常可能是传感器故障或控制面板问题。已为您登记报修。冰箱保修期一般为购买后12个月，具体需根据您的购买日期和产品序列号查询。请提供购买日期或序列号以便确认保修状态。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-098", "composite",
    "咖啡机需要除垢了，帮我查一下除垢步骤，另外我上次买的除垢剂用完了，想再订一盒",
    ["除垢", "0.5L", "订购"],
    "除垢步骤：取出胶囊并关闭手柄，水箱加入0.5L饮用水和1份除垢液，同时按Espresso和Lungo键3秒进入除垢模式，按Lungo键运行直到水箱变空，用收集的除垢液重复一次，再用清水冲洗一次。关于订购除垢剂，建议联系Nespresso Club购买原厂除垢套件。",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["troubleshooting", "product_support", "case_intake"])

add("qa-099", "composite",
    "洗衣机洗完衣服后有异味，而且我注意到最近脱水的时候声音比以前大很多",
    ["异味", "声音", "清洁"],
    "两个问题分别处理：异味问题——建议清洁洗衣桶内部（注水至O水位后运行洗涤1-2分钟再排水），并清洁棉绒过滤器。噪声增大——检查洗衣机是否放置在平整地面（允许2°倾斜），衣物是否在脱水桶中均匀分布，底脚是否牢固。如问题持续需技术人员检查。",
    "Washing_Machine.txt", "Washing Machine",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-100", "composite",
    "昨天买的VR头显，感觉有点漏光而且遮光罩戴着不太舒服，这个可以调节吗？",
    ["漏光", "遮光罩", "调节"],
    "遮光罩可从VR头显上取下用水清洗后重新安装（将遮光罩上的凸点与头显上的13个孔对齐即可）。佩戴不适可调节头带和镜罩位置——使用调节旋钮调整松紧度，但不要拧得过紧。镜罩位置可调节以适应戴眼镜用户。建议按照系统功能界面中的佩戴说明进行调整。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-101", "composite",
    "烤箱定时功能不会用，想设置2小时后自动开始烤，烤1小时。另外烤箱里面照明灯太暗了能换亮一点的吗？",
    ["定时", "自动", "灯泡", "25-40W"],
    "设置延迟烹饪：先设置烹饪时间为1小时，再按中间按钮显示End和烹饪结束时间，按加号按钮设置烹饪结束时间（最长延迟23小时59分钟）。关于照明灯，烤箱灯泡为25-40W/230V E-14、耐温300°C白炽灯，这是标准配件不可更换为更高亮度，否则可能超出耐温范围。更换灯泡前请断开电源。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["product_support", "troubleshooting"])

add("qa-102", "composite",
    "My airfryer makes a weird noise during cooking and the keep warm function doesn't seem to work properly. Help!",
    ["noise", "keep warm", "basket"],
    "Regarding the noise: check that pan and basket are properly inserted. Food residues on the heating element can cause noise — clean with soft to medium bristle brush after cooling. For keep warm: press menu button until keep warm icon blinks, then press On/Off to start. Default timer is 30 minutes, adjustable 1-30 min. Temperature cannot be changed in keep warm mode. The fan and heater turn on intermittently during keep warm — this is normal.",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["troubleshooting", "product_support"], language="en")

add("qa-103", "composite",
    "吹风机怠速不稳还容易熄火，火花塞也刚换的，不知道是不是化油器问题。顺便帮我查一下我的保修期",
    ["怠速", "化油器", "火花塞", "保修"],
    "怠速不稳和熄火问题：1) 确认火花塞电极间隙为0.5mm，使用推荐型号火花塞；2) 调节化油器T螺钉使发动机正确怠速，转速不偏高；3) 微调L螺钉使发动机达到最大怠速转速，然后将L螺钉逆时针旋出最多1/4圈；4) 检查空气滤清器是否需要清洁。关于保修期查询，请提供购买日期和产品序列号。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-104", "composite",
    "微波炉加热不均匀，有的地方烫有的地方凉的，还有显示屏上的时间一直在闪怎么解决？",
    ["加热不均", "转盘", "闪烁", "时钟"],
    "加热不均匀：确保转盘功能已开启（不在OFF状态），将食物单层排列，密度较大的食物放在外侧，烹饪中途翻转食物。时间闪烁：说明时钟未设置或断电后需重新设置当前时间。按Clock键进入时间设置模式输入正确时间即可。",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-105", "composite",
    "蓝牙鼠标光标跳来跳去不精准，而且是新换的电池。另外这个鼠标能同时连接两台电脑吗？",
    ["光标", "精准", "配对", "两台"],
    "光标不精准：1) 请在干净、平整、不光滑的表面使用鼠标，2) 检查鼠标垫是否为深色（深色垫会增加功耗），3) 确认传感器窗口清洁无遮挡。关于连接多设备：蓝牙鼠标一次只能与一台设备配对，如需切换设备需重新配对。LED绿色闪烁表示鼠标正在配对中。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-106", "composite",
    "洗碗机洗完后碗碟上有白色水渍，而且最近耗盐量好像特别快。这是什么问题？",
    ["白色水渍", "耗盐", "亮碟剂"],
    "白色水渍可能原因：1) 亮碟剂（漂洗剂）不足，请检查亮碟剂分配器并补充；2) 水质硬度设置不当导致用盐量过大。耗盐过快可能是软化器设置的水硬度等级偏高，请根据当地水质硬度调整设置。同时检查盐仓盖是否拧紧，未密封会导致盐水过量消耗。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-107", "composite",
    "空调遥控器找不到放哪了，手机上能控制吗？另外空调定时功能半夜自己取消了是怎么回事？",
    ["手机", "定时", "取消"],
    "关于手机控制：部分空调型号支持WiFi连接和手机App控制，请确认您的空调是否具备此功能并已连接家庭WiFi。定时取消：可能是定时设置后发生过断电导致设置丢失，也可能是设置了单次定时而非每日定时。建议重新设置并确认选择的是每日重复模式。",
    "空调手册.txt", "空调",
    acceptable_skills=["product_support", "troubleshooting"])

add("qa-108", "composite",
    "The Nespresso machine is not recognizing capsules and also the coffee is coming out cold. What's going on?",
    ["capsule", "cold", "heating"],
    "If machine doesn't recognize capsule: ensure lever is completely closed. Never use damaged or deformed capsules. If capsule is stuck, turn off and unplug before checking. For cold coffee: machine needs about 25 seconds to heat up (blinking lights = heating, steady = ready). If coffee is still cold after machine indicates ready, the heating element may need service. Contact Nespresso Club for assistance.",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["troubleshooting", "product_support"], language="en")

add("qa-109", "composite",
    "空气净化器睡眠模式风噪比之前大了，而且滤网更换指示灯也不亮了。怎么判断滤网该不该换？",
    ["滤网", "指示灯", "睡眠模式"],
    "睡眠模式风噪增大：检查进风口是否有异物堵塞，滤网是否积灰过多导致风扇负荷增大。滤网指示灯不亮可能是灯珠故障，不能仅依赖指示灯判断。判断滤网是否需更换：查看滤网颜色（从白色变灰黑需更换），使用时间超过推荐周期（HEPA一般6-12个月），或出风量明显下降。",
    "空气净化器手册.txt", "空气净化器",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-110", "composite",
    "VR头显玩了一段时间后追踪不准了，而且耳机里有持续的电流声。需要怎么解决？",
    ["追踪", "电流声", "摄像头"],
    "追踪不准：1) 按照系统快捷菜单中的[调整VR]>[确认位置]调整摄像头位置及角度，2) 检查游玩区域光线是否充足且VR头显追踪灯无遮挡，3) 确保在游玩区域中央操作。耳机电流声：在干燥环境下耳朵可能偶尔感到轻微静电触击，这是人体静电积累而非设备故障。若持续存在，请检查耳机接头是否清洁无灰尘，并确认音频线缆完好。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["troubleshooting", "product_support"])

# ============ boundary (15 items) ============

add("qa-111", "boundary",
    "？",  # 极短消息
    ["无法理解", "帮助"],
    "您好，我没有完全理解您的问题。请问有什么可以帮助您的？您可以直接描述您遇到的设备问题，我会尽力帮您解答。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["general"])

add("qa-112", "boundary",
    "哈哈哈哈哈哈哈哈哈哈哈哈哈",
    ["帮助", "问题"],
    "您好！请问有什么可以帮您解决的问题吗？请描述您使用的产品和遇到的问题，我会为您提供帮助。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["general"])

add("qa-113", "boundary",
    "a" * 2000,  # 极长消息
    ["帮助", "理解"],
    "您好，您的消息很长但我注意到主要是重复的字符。请问您具体想咨询什么问题？请用简洁的语言描述您的产品或服务需求，我会尽力帮助您。",
    "空调手册.txt", "空调",
    acceptable_skills=["general"])

add("qa-114", "boundary",
    "帮我查一下产品型号为AC900-B2 Pro Max Ultra 2024冬季限量版的问题",
    ["产品", "型号", "查询"],
    "您提到的产品型号'AC900-B2 Pro Max Ultra 2024冬季限量版'不在我们的标准产品数据库中。可能是型号输入有误或属于限量特殊版本。请确认产品铭牌上的准确型号（通常位于产品背面或底部），或提供产品序列号以便查询。",
    "空调手册.txt", "AC900",
    acceptable_skills=["product_support", "general"])

add("qa-115", "boundary",
    "Please help me in French, German, and Japanese all at once",
    ["language", "support"],
    "I can assist you, but I work best in one language at a time. Let me help you in English first. What product issue can I help you with today?",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["general"], language="en")

add("qa-116", "boundary",
    "关机吗",
    ["关机", "帮助"],
    "您好，我没有完全理解您的问题。请问您是想了解如何关闭设备吗？如果是，请告诉我您使用的产品型号，我可以为您提供该设备的关机操作说明。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["general"])

add("qa-117", "boundary",
    "",  # 空消息 - 系统会拒绝，但这是测试边界
    [],
    "",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["general"])

add("qa-118", "boundary",
    "🐛🐛🐛🔥🔥🔥",
    ["理解", "问题"],
    "您好，我无法从您发送的表情符号中理解具体问题。请用文字描述您遇到的问题，告诉我您使用的产品和具体情况，我会为您提供帮助。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["general"])

add("qa-119", "boundary",
    "我家的空调冰箱洗衣机洗碗机烤箱微波炉吹风机咖啡机全都坏了怎么办",
    ["多个", "逐一", "产品"],
    "您提到了多种产品出现问题，建议我们逐一排查。请先描述最紧急的一个产品的具体故障现象（例如：是什么产品、什么症状、什么时候开始的）。我会逐件帮您分析处理。",
    "空调手册.txt", "多产品",
    acceptable_skills=["general", "case_intake"])

add("qa-120", "boundary",
    "What's the meaning of life? Also, how do I clean my airfryer?",
    ["irrelevant", "clean", "airfryer"],
    "I'll focus on what I can help with — cleaning your Airfryer. Let the basket, pan, and inside of the appliance cool completely before cleaning. The pan and basket can be cleaned in a dishwasher or with hot water, dishwashing liquid and a non-abrasive sponge. Do not use metal kitchen utensils or abrasive cleaning materials. Clean after every use and dispose of rendered fat or oil from the bottom of the pan.",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["product_support", "general"], language="en")

add("qa-121", "boundary",
    "帮我写一篇关于空调维修的5000字论文",
    ["帮助", "问题"],
    "您好，我是产品售后服务助手，专注于帮助解决产品使用问题和售后需求。我可以帮您解答空调的故障排查、使用操作、维修报修等实际问题。请告诉我您具体遇到了什么产品问题？",
    "空调手册.txt", "空调",
    acceptable_skills=["general"])

add("qa-122", "boundary",
    "User manual says 'E2 error code' but my model doesn't have error codes. Model is AC900.",
    ["E2", "manual", "check"],
    "If your AC900 manual mentions the E2 error code but your unit doesn't display it, the manual may cover multiple models with different features. Check the model-specific section or the troubleshooting chapter for AC900-specific information. Error code display availability varies by sub-model configuration.",
    "空调手册.txt", "AC900",
    acceptable_skills=["product_support", "troubleshooting"], language="en")

add("qa-123", "boundary",
    "我想知道如何同时使用空气炸锅的8个预设程序来准备一顿完整的晚餐，包括前菜、主菜和甜点，还要保温",
    ["预设", "保温", "批量"],
    "空气炸锅支持多个预设程序，但一次只能运行一个烹饪程序。要准备完整晚餐建议分批烹饪：1) 先用冷冻零食预设制作前菜，2) 用对应预设烹饪主菜，3) 最后制作甜点。每批完成后空气炸锅可立即开始下一批。烹饪完成后可使用保温模式（默认30分钟，1-30分钟可调）保持食物温度。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["product_support"])

add("qa-124", "boundary",
    "我今年85岁了，第一次用智能手机和APP来控制烤箱，完全听不懂术语，能一步一步教我",
    ["简单", "步骤", "一步一步"],
    "好的，我用最简单的方式一步一步教您：\n第一步：插上烤箱的电源插头。\n第二步：烤箱屏幕上会出现数字和符号。\n第三步：烤箱上面有一个圆形的旋钮，您转到有风扇图标的那个位置。\n第四步：再转另一个旋钮，转到有180这个数字的位置。\n第五步：等一会儿烤箱就会变热了。\n如果您觉得复杂，可以让家里年轻人帮您先设置好时间和温度，您只需要把食物放进去就行。有任何不明白的随时问我，不用着急。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["product_support", "general"])

add("qa-125", "boundary",
    "The temperature was set to 180°C but my food burned. The manual says cooking time should be 20 minutes. I set exactly 20 minutes. Why did it burn?",
    ["temperature", "time", "preheat", "rack position"],
    "Several factors could cause burning even with correct time/temperature: 1) Preheating — the 20-minute cooking time assumes food is placed in after the oven reaches temperature. If you included preheating time, food was in too long. 2) Rack position — food placed too close to the heating element will burn faster. Use lower rack positions. 3) Dark metal pans absorb more heat than light ones. 4) Food quantity — smaller portions cook faster. Try reducing temperature by 10-20°C for your specific oven.",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["troubleshooting", "product_support"], language="en")

# ============ multi-turn (10 items) ============

add("qa-126", "multi-turn",
    "我的空气炸锅有问题",
    ["问题", "帮助"],
    "您好，请告诉我您的空气炸锅具体出现了什么问题？例如：不加热、冒烟、显示屏异常、烹饪效果不佳等。请尽可能详细描述故障现象。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["product_support", "troubleshooting"])

add("qa-127", "multi-turn",
    "上次报修的空调，维修师傅来了说修好了，但用了两天又不制冷了",
    ["再次故障", "维修", "不制冷"],
    "很抱歉维修后再次出现同样问题。这说明之前的维修可能未解决根本原因。我已调出您之前的维修记录。我们将安排高级技师重新上门检修，更换的配件和人工不另收费（属于维修保修范围）。同时我们将把此案例标记为重复维修跟进。",
    "空调手册.txt", "空调",
    acceptable_skills=["case_intake", "troubleshooting"])

add("qa-128", "multi-turn",
    "I followed your cleaning instructions for the microwave but the smell still hasn't gone away. What else can I do?",
    ["odor", "charcoal filter", "vent"],
    "If the smell persists after cleaning, check the charcoal filter. If your microwave is vented inside, the charcoal filter should be replaced every 6-12 months and cannot be cleaned. Also check the grease filter (clean at least monthly). Run the vent fan on HI for 30 minutes after cooking. Place a bowl of water with lemon juice in the microwave and heat for 3 minutes — the steam helps neutralize odors.",
    "Microwave_OTR.txt", "Microwave OTR",
    acceptable_skills=["troubleshooting", "product_support"], language="en")

add("qa-129", "multi-turn",
    "上次你说洗碗机漏水是门封条的问题，我换了门封条但还是漏，而且漏得更多了",
    ["门封条", "更换", "仍然漏水"],
    "更换门封条后漏水加重，可能原因：1) 门封条安装不到位或型号不匹配，2) 漏水源头并非门封条而是其他部件（如进水阀、排水管连接处、内胆密封），3) 门铰链松动导致关门后密封不严。建议停止使用并安排专业技术人员上门检测确认真正漏水源。请勿再自行维修以免影响保修。",
    "洗碗机手册.txt", "洗碗机",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-130", "multi-turn",
    "按照你说的换了新电池也重新配对了，蓝牙鼠标还是连不上",
    ["新电池", "配对", "连不上"],
    "已更换电池且重新配对仍无法连接时，请尝试以下深度排查：1) 到控制面板>鼠标>蓝牙查看鼠标电量状态，2) 卸载并重新安装WIDCOMM蓝牙驱动程序，3) 确认电脑蓝牙功能未被禁用，4) 使用USB蓝牙接收器（而非电脑自带蓝牙）尝试配对。若以上均无效，鼠标可能存在硬件故障，可联系经销商获取技术支持或申请保修。",
    "蓝牙激光鼠标手册.txt", "蓝牙激光鼠标",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-131", "multi-turn",
    "咖啡机除垢后还是有苦涩味，而且出咖啡速度变慢了",
    ["苦涩味", "速度慢", "清洁"],
    "除垢后仍有苦涩味和出咖啡变慢：1) 确保已用清水充分冲洗机器（可能需要多冲洗一个循环），2) 苦涩味可能来自咖啡胶囊本身而非机器，请检查胶囊保质期，3) 出咖啡变慢可能出口堵塞，用软湿布清洁咖啡出口，4) 检查水箱是否使用了新鲜饮用水（非存放多日的水）。如问题持续请联系Nespresso Club。",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["troubleshooting", "product_support"])

add("qa-132", "multi-turn",
    "上次你教我清洁了空调滤网，制冷好了一点但还是不够冷，还有别的原因吗？",
    ["滤网", "不够冷", "其他原因"],
    "滤网清洁后制冷仍不足可能是以下原因：1) 制冷剂不足或泄漏（需专业人员检测），2) 室外机散热不良（周围有遮挡物或冷凝器脏污），3) 房间面积超过空调制冷能力，4) 温度传感器故障导致压缩机未正常工作。建议安排技术人员上门检测制冷剂压力和系统运行参数。",
    "空调手册.txt", "空调",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-133", "multi-turn",
    "冰箱温度还是不稳定，上次让我检查的门封条我检查了没问题",
    ["温度不稳定", "门封条", "继续排查"],
    "门封条正常但温度仍不稳定时，建议排查：1) 冰箱是否频繁开门或门未关严，2) 是否放入大量未冷却的热食导致箱内温度升高，3) 冷凝器盘管是否积灰影响散热，4) 温控器或温度传感器是否异常，5) 冰箱周围通风空间是否足够（背面和侧面需要散热空间）。如均正常仍需安排技术人员上门检测制冷系统。",
    "冰箱手册.txt", "冰箱",
    acceptable_skills=["troubleshooting", "case_intake"])

add("qa-134", "multi-turn",
    "之前关于VR头晕你说适应一下就好了，我已经试了两周了还是不行。还有其他办法吗？",
    ["头晕", "两周", "适应"],
    "适应两周后仍持续出现VR眩晕，建议尝试：1) 从短时间（5-10分钟）开始，逐步延长，2) 优先选择低运动强度的VR内容（如360度视频而非第一人称动作游戏），3) 确保游玩区域光线充足、通风良好，4) 调整瞳距设置使镜片与眼睛距离合适，5) 尝试使用晕车药（请先咨询医生）。如果所有方法都无效，此产品可能不适合您的体质，可考虑联系购买渠道了解退货选项。",
    "VR头显手册.txt", "VR头显",
    acceptable_skills=["troubleshooting", "product_support", "refund"])

add("qa-135", "multi-turn",
    "吹风机按你说的清洗了空气滤清器也调了化油器，现在怠速不熄火了但是高转速没力",
    ["化油器", "高转速", "无力"],
    "怠速改善但高转速无力，调节方法：1) 全开油门，调节H油针直至发动机达到最大转速，2) 将H油针逆时针旋1/8圈（若限位限制则少于1/8圈）。注意：调节H油针时全油门持续时间最长10秒，之后需怠速至少10秒。另外检查燃油混合是否正确（1:50），机油过多会导致动力下降和火花塞积碳。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["troubleshooting"])

# ============ general (10 items) ============

add("qa-136", "general",
    "你好，我想了解一下你们的产品保修政策",
    ["保修", "12个月", "保修范围"],
    "我们的产品保修政策：自购买之日起12个月内，对因材料或工艺缺陷导致的故障提供免费维修服务。保修需在正常家庭使用条件下有效。以下情况不在保修范围：商业用途、不当使用、未经授权的维修、外观损坏、自然灾害造成的损坏等。具体保修条款请以产品随附的保修手册为准。",
    "Microwave_OTR.txt", "多产品",
    acceptable_skills=["general"])

add("qa-137", "general",
    "What products do you sell?",
    ["product", "appliance"],
    "I'm a product support assistant here to help with issues related to home appliances and electronics you've purchased. I can assist with troubleshooting, usage questions, warranty claims, and repairs. To help you better, please tell me which product you need help with.",
    "Airfryer.txt", "多产品",
    acceptable_skills=["general"], language="en")

add("qa-138", "general",
    "你们公司叫什么名字？",
    ["公司", "品牌"],
    "您好！我是您的产品售后助手，专注于帮您解决产品的使用、故障、维修和售后问题。关于公司/品牌具体信息，建议您查看产品包装或用户手册中的品牌信息，或访问官方网站了解更多。请问有什么产品问题我可以帮您解决？",
    "洗碗机手册.txt", "多产品",
    acceptable_skills=["general"])

add("qa-139", "general",
    "怎么联系客服电话？",
    ["客服", "电话", "联系"],
    "您可以通过以下方式联系客服：产品保修手册中注明的售后服务中心电话、官方网站在线客服、或前往购买门店咨询。如需技术支持，我可帮您在线解决大部分常见问题。请问您遇到了什么产品问题？",
    "烤箱手册.txt", "多产品",
    acceptable_skills=["general", "handoff"])

add("qa-140", "general",
    "What are your business hours?",
    ["hours", "support"],
    "I'm an automated product support assistant available 24/7 to help with your product questions. For live agent availability and business hours in your region, please check your product's warranty manual or contact your local customer service center. How can I help you with your product today?",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["general"], language="en")

add("qa-141", "general",
    "今天天气真好",
    ["帮助", "问题"],
    "今天的天气确实不错！不过我是您的产品售后助手，主要帮您解决产品使用和售后问题。如果您的家电或电子产品有任何问题需要帮助，请随时告诉我。",
    "空调手册.txt", "多产品",
    acceptable_skills=["general"])

add("qa-142", "general",
    "能不能给我推荐一款好用的洗碗机？",
    ["推荐", "产品"],
    "我是售后支持助手，主要帮助解决已购买产品的使用和维修问题，无法提供产品推荐或销售建议。如需选购产品，建议您访问官方网站查看各型号功能和用户评价，或前往实体店咨询销售顾问。如果您已拥有我们的产品需要帮助，我很乐意为您服务。",
    "洗碗机手册.txt", "多产品",
    acceptable_skills=["general"])

add("qa-143", "general",
    "Do you have a store near me?",
    ["store", "location"],
    "I'm an online product support assistant and don't have access to store location data. For store locations, please visit the official website's store locator or check your product's warranty documentation for authorized service centers in your area. Is there a product issue I can help you with?",
    "Microwave_OTR.txt", "多产品",
    acceptable_skills=["general"], language="en")

add("qa-144", "general",
    "我想了解一下你们有没有以旧换新的政策",
    ["以旧换新", "回收"],
    "关于以旧换新政策，不同产品和地区政策可能有所不同，建议咨询当地授权经销商或访问官方网站了解最新以旧换新和回收方案。产品报废处理方面，请按照当地法规进行处置，电气电子产品不应作为普通生活垃圾处理。如需帮助解决现有产品的问题，我随时为您服务。",
    "烤箱手册.txt", "多产品",
    acceptable_skills=["general"])

add("qa-145", "general",
    "帮我查一下订单物流到哪里了 ORD-88888",
    ["物流", "ORD-88888"],
    "我无法直接查询物流信息，但我可以帮您确认订单号ORD-88888的信息。关于物流配送状态，建议您通过购买平台的订单详情页查看实时物流信息，或联系购买渠道的客服获取准确的配送状态。如果您收到产品后有任何使用问题，我随时为您提供技术支持。",
    "冰箱手册.txt", "多产品",
    acceptable_skills=["general", "case_intake"])

# ============ no_evidence (5 items) ============

add("qa-146", "no_evidence",
    "空气炸锅的蓝牙功能怎么连接？我的手机搜索不到设备",
    ["不支持", "无线"],
    "根据空气炸锅手册，该产品支持WiFi连接功能（2.4 GHz 802.11 b/g/n）和NutriU App控制，但未提及蓝牙功能。请确认您要通过WiFi还是蓝牙连接？如需连接手机请使用NutriU App通过WiFi进行配对和控制。",
    "Airfryer.txt", "Airfryer",
    acceptable_skills=["product_support"])

add("qa-147", "no_evidence",
    "烤箱有微波功能吗？我想用来热牛奶",
    ["烤箱", "微波", "不支持"],
    "根据烤箱用户手册，本产品为传统电烤箱，不具备微波加热功能。烤箱设有上下加热管、热风循环等功能，适合烘焙、烤制等烹饪方式。热牛奶建议使用微波炉或明火加热。请勿用烤箱加热密封容器中的液体。",
    "烤箱手册.txt", "烤箱",
    acceptable_skills=["product_support", "general"])

add("qa-148", "no_evidence",
    "Does the Nespresso machine make iced coffee?",
    ["iced", "hot", "cold brew"],
    "The Nespresso machine is designed to brew hot espresso (40ml) and lungo (110ml) coffee. The user manual does not describe an iced coffee or cold brew function. However, you can brew espresso over ice manually — brew a shot and pour it over a cup filled with ice for a refreshing iced espresso drink.",
    "Coffee_Machine.txt", "Nespresso",
    acceptable_skills=["product_support", "general"], language="en")

add("qa-149", "no_evidence",
    "洗衣机能烘干吗？我洗完衣服想直接烘干",
    ["烘干", "不支持"],
    "根据洗衣机用户手册，本产品为双桶洗衣机，具备洗涤和脱水功能，但不具备烘干功能。脱水后的衣物需要自然晾晒或用独立烘干机烘干。手册中脱水定时器设置为1-5分钟（部分型号1-10分钟）。",
    "Washing_Machine.txt", "Washing Machine",
    acceptable_skills=["product_support"])

add("qa-150", "no_evidence",
    "吹风机可以用来吹头发吗？",
    ["吹头发", "地面", "落叶"],
    "本产品为225B型户外吹风机/吹叶机，设计用于吹扫地面落叶及杂物，并非美发用吹风机。该产品使用二冲程汽油发动机驱动，操作时需佩戴听力防护、眼部防护等专业防护装备，禁止在室内使用。请勿将本产品用于吹头发等非设计用途。",
    "吹风机手册.txt", "225B吹风机",
    acceptable_skills=["product_support", "troubleshooting"])


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in ITEMS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ITEMS)} items to {OUTPUT}")

    # 统计类别分布
    from collections import Counter
    cats = Counter(i["category"] for i in ITEMS)
    print("\nCategory distribution:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"  TOTAL: {sum(cats.values())}")
