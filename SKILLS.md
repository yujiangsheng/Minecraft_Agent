# Luanti AI 智能体 — 动作与技能体系

> 作者: Jiangsheng Yu | 许可证: MIT License
>
> 本文档描述智能体支持的 **36 种原子动作**、**30+ 种兼容别名**、
> **14 种高级复合技能** 以及 **完整的状态感知字段**。

---

## 一、原子动作清单（36 种）

### 基础移动（8 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `explore` | 向随机方向移动探索 | — |
| `move_to` | 移动到指定坐标 | `{x, y, z}` |
| `jump` | 跳跃（翻越障碍） | — |
| `retreat` | 向后撤退 | — |
| `flee_from` | 高速远离方向逃跑 | — |
| `swim` | 水中游泳（含上浮） | — |
| `sneak` | 开启/关闭潜行 | `{enable: true/false}` |
| `sprint` | 开启/关闭冲刺 | `{enable: true/false, speed: 1.5}` |

### 视角控制（2 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `look_around` | 原地观察（刷新感知） | — |
| `look_at` | 注视指定位置 | `{target: {x,y,z}}` 或 `{yaw, pitch}` |

### 采集/挖掘（7 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `gather_wood` | 砍伐附近树木 | — |
| `mine_stone` | 开采附近石头 | — |
| `mine_ore` | 开采矿石 | `{ore_type: "iron_ore\|coal\|..."}` |
| `dig` | 挖掘面前方块 | `{target_type: "tree\|stone\|ore"}` |
| `dig_down` | 向下挖掘 | `{depth: 1-5}` |
| `dig_up` | 向上挖掘 | `{height: 1-3}` |
| `tunnel` | 向前挖 2 格高通道 | `{length: 1-8}` |

### 放置/建造（6 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `place_block` | 放置方块到指定坐标 | `{node, target: {x,y,z}}` |
| `place_at` | 智能相对放置 | `{node, direction: "front\|back\|..."}` |
| `build_shelter` | 建造 3×3×3 庇护所 | `{material}` |
| `bridge` | 向前搭桥 | `{length: 1-10, block}` |
| `tower_up` | 垂直搭建柱子 | `{height: 1-10, block}` |
| `light_area` | 放置火把照明 | — |

### 物品管理（4 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `equip` | 装备物品 | `{item}` |
| `drop_item` | 丢弃物品 | `{item, count}` |
| `set_hotbar` | 切换快捷栏位 | `{slot}` |
| `sort_inventory` | 整理库存 | — |

### 合成/冶炼（4 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `craft_tool` | 合成工具 | `{item}` |
| `craft_item` | 合成物品 | `{item}` |
| `smelt` | 熔炉冶炼 | `{item}` |
| `check_recipe` | 查询配方 | `{item}` |

### 进食/战斗（4 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `eat_food` | 进食恢复血量 | `{item}` |
| `attack` | 攻击最近敌对实体 | — |
| `attack_entity` | 攻击指定类型实体 | `{target}` |
| `punch_node` | 敲击方块 | — |

### 容器交互（3 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `deposit_item` | 存入箱子 | `{item}` |
| `take_from_container` | 取出箱子 | `{item}` |
| `use_node` | 右键交互 | — |

### 搜索/感知（2 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `find_resource` | 搜索附近资源 | `{resource}` |
| `pickup_item` | 拾取掉落物 | — |

### 农业（2 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `farm_plant` | 种植 | `{seed}` |
| `farm_harvest` | 收获 | — |

### 等待（1 种）

| 动作 | 说明 | 参数 |
|------|------|------|
| `wait` | 原地等待 | `{duration}` |

---

## 二、兼容别名映射（30+ 种）

LLM 可能生成非标准动作名称，以下别名会自动映射到对应核心动作：

| 别名 | 映射到 | 别名 | 映射到 |
|------|--------|------|--------|
| `move` | `move_to` | `flee` / `run_away` | `retreat` |
| `chop_tree` / `break_log` | `dig(tree)` | `mine` / `collect_stone` | `dig(stone)` |
| `collect_wood` | `dig(tree)` | `find_tree` | `find_resource(tree)` |
| `gather_food` | `find_resource(food)` | `hunt_food` | `attack` |
| `assess_situation` | `look_around` | `find_safe_spot` | `move_to` |
| `build` / `hide` | `build_shelter` | `place` | `place_block` |
| `rest` | `wait(3s)` | `sleep` | `wait(5s)` |
| `crouch` | `sneak(on)` | `stop_sneak` | `sneak(off)` |
| `plant` | `farm_plant` | `harvest` | `farm_harvest` |
| `cook` / `furnace` | `smelt` | `drop` / `discard` | `drop_item` |
| `open_chest` / `interact` / `use` | `use_node` | `pick_up` / `collect` | `pickup_item` |
| `organize_inventory` | `sort_inventory` | `dig_forward` | `tunnel` |
| `pillar_up` / `scaffold` | `tower_up` | — | — |

---

## 三、高级复合技能（14 种）

以下技能是由多个原子动作组合而成的高级行为，可通过 `action_plan` 编排：

| # | 技能名 | 动作序列 | 触发条件 |
|---|--------|----------|----------|
| 1 | 伐木收集 | `gather_wood → pickup_item → sort_inventory` | 木材 < 10 |
| 2 | 采矿远征 | `find_resource → equip(pickaxe) → dig → pickup_item` | 需要石头 |
| 3 | 冶铁流程 | `find_resource(iron) → mine_ore → find_resource(coal) → mine_ore → smelt` | 需要铁锭 |
| 4 | 建造房屋 | `gather_wood → craft_item(planks) → build_shelter` | 夜晚将至且无庇护所 |
| 5 | 紧急逃生 | `sprint → retreat → jump → build_shelter(dirt)` | HP < 5 且有敌怪 |
| 6 | 挖矿下探 | `dig_down(3) → light_area → dig_down(3) → light_area` | 需要深层矿石 |
| 7 | 桥梁穿越 | `bridge(6, cobble) → move_to` | 前方有间隙/水域 |
| 8 | 垂直攀升 | `tower_up(5) → look_around` | 需要侦察登高 |
| 9 | 隧道挖掘 | `equip(pickaxe) → tunnel(5) → light_area` | 需要穿越山体 |
| 10 | 农业循环 | `farm_plant → wait(300) → farm_harvest → farm_plant` | 有种子和耕地 |
| 11 | 物资整理 | `sort_inventory → deposit_item → take_from_container` | 库存接近满 |
| 12 | 夜间防御 | `build_shelter → light_area → equip(sword) → look_around → wait` | 夜晚降临 |
| 13 | 水下探索 | `swim → dig → swim → pickup_item` | 水中有资源 |
| 14 | 合成升级 | `check_recipe → gather_wood → craft_item(sticks) → craft_tool` | 需要更好工具 |

---

## 四、状态感知字段

智能体每个 tick 收到的完整游戏状态 JSON：

| 字段 | 类型 | 说明 |
|------|------|------|
| `time` | `"day"/"night"/"dawn"/"dusk"` | 时间阶段 |
| `time_of_day` | `0.0 - 1.0` | 精确时间 |
| `day_count` | `int` | 已过天数 |
| `health` | `0 - 20` | 当前生命值 |
| `max_health` | `20` | 最大生命值 |
| `hunger` | `0 - 20` | 饥饿值 |
| `breath` | `0 - 10` | 呼吸值（水下消耗）|
| `position` | `{x, y, z}` | 玩家位置 |
| `look_dir` | `{x, y, z}` | 视线方向向量 |
| `inventory` | `{item: count, ...}` | 库存（去掉 mod 前缀）|
| `nearby_entities` | `[{type, hostile, distance, health}, ...]` | 附近实体 |
| `nearby_blocks` | `[{name, count}, ...]` | 附近方块（前 20 种）|
| `has_shelter` | `bool` | 头顶是否有遮蔽 |
| `light_level` | `0 - 15` | 当前光照等级 |
| `biome_name` | `string` | 当前生物群系 |
| `has_weapon` | `bool` | 是否持有武器 |
