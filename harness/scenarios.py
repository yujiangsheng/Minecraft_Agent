"""
Harness Scenarios — 标准化场景库

定义 50+ 种游戏状态快照，用于各评测模块的输入。
每个场景包含：环境状态、期望行为标签、难度等级。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import json
import copy


@dataclass
class Scenario:
    """评测场景"""
    id: str
    name: str
    description: str
    category: str                          # survival / combat / resource / craft / explore
    difficulty: str                        # easy / medium / hard
    env_state: Dict[str, Any]              # 环境状态快照
    acceptable_actions: List[str]          # 合理动作集合（任取其一即可）
    acceptable_modes: List[str]            # 合理模式集合
    unacceptable_actions: List[str] = field(default_factory=list)   # 明确不合理的动作
    expected_risk_level: str = ""          # 期望风险等级 low/medium/high
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "env_state": self.env_state,
            "acceptable_actions": self.acceptable_actions,
            "acceptable_modes": self.acceptable_modes,
            "unacceptable_actions": self.unacceptable_actions,
            "expected_risk_level": self.expected_risk_level,
            "tags": self.tags,
        }


class ScenarioLibrary:
    """场景库管理器"""

    def __init__(self):
        self.scenarios: List[Scenario] = []
        self._load_builtin()

    # ════════════════════════════════════════════
    #  查询
    # ════════════════════════════════════════════

    def get(self, scenario_id: str) -> Optional[Scenario]:
        for s in self.scenarios:
            if s.id == scenario_id:
                return s
        return None

    def filter(self, category: str = None, difficulty: str = None,
               tags: List[str] = None) -> List[Scenario]:
        result = self.scenarios
        if category:
            result = [s for s in result if s.category == category]
        if difficulty:
            result = [s for s in result if s.difficulty == difficulty]
        if tags:
            tag_set = set(tags)
            result = [s for s in result if tag_set & set(s.tags)]
        return result

    def all(self) -> List[Scenario]:
        return list(self.scenarios)

    def add(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.scenarios.append(Scenario(**item))

    def save_to_file(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in self.scenarios], f,
                      ensure_ascii=False, indent=2)

    # ════════════════════════════════════════════
    #  内置 50+ 场景
    # ════════════════════════════════════════════

    def _load_builtin(self):
        # ── 基础状态模板 ──
        base = {
            "time": "day",
            "health": 20,
            "hunger": 20,
            "position": {"x": 100, "y": 10, "z": 100},
            "inventory": {},
            "nearby_entities": [],
            "nearby_blocks": ["tree", "dirt", "stone"],
            "has_shelter": False,
        }

        def s(overrides: dict) -> dict:
            """快速合并生成 env_state"""
            state = copy.deepcopy(base)
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(state.get(k), dict):
                    state[k].update(v)
                else:
                    state[k] = v
            return state

        scenarios = [
            # ═══════════ 生存 — 紧急 ═══════════
            Scenario(
                id="surv_01", name="濒死无食物",
                description="生命值仅剩 2，库存无食物，白天",
                category="survival", difficulty="hard",
                env_state=s({"health": 2, "hunger": 5}),
                acceptable_actions=["eat_food", "find_resource", "retreat", "flee_from"],
                acceptable_modes=["survive", "retreat"],
                unacceptable_actions=["explore", "attack", "mine_stone", "tunnel", "bridge"],
                expected_risk_level="high",
                tags=["critical_health", "no_food"],
            ),
            Scenario(
                id="surv_02", name="濒死有食物",
                description="生命值 3，库存有苹果，白天",
                category="survival", difficulty="medium",
                env_state=s({"health": 3, "inventory": {"apple": 3}}),
                acceptable_actions=["eat_food"],
                acceptable_modes=["survive"],
                unacceptable_actions=["explore", "attack", "mine_stone", "dig"],
                expected_risk_level="high",
                tags=["critical_health", "has_food"],
            ),
            Scenario(
                id="surv_03", name="饥饿危急",
                description="饥饿值 2，生命正常，附近有被动生物",
                category="survival", difficulty="medium",
                env_state=s({
                    "hunger": 2,
                    "nearby_entities": [{"type": "cow", "hostile": False, "distance": 5}],
                }),
                acceptable_actions=["eat_food", "find_resource", "attack_entity", "attack", "farm_harvest"],
                acceptable_modes=["survive", "gather"],
                unacceptable_actions=["tunnel", "bridge", "build_shelter"],
                expected_risk_level="medium",
                tags=["critical_hunger"],
            ),
            Scenario(
                id="surv_04", name="夜间无庇护所",
                description="夜晚，无庇护所，无敌人，有木材",
                category="survival", difficulty="medium",
                env_state=s({
                    "time": "night",
                    "inventory": {"wood": 20, "stick": 4},
                    "nearby_blocks": ["dirt", "stone"],
                }),
                acceptable_actions=["build_shelter", "light_area", "place_block", "place_at"],
                acceptable_modes=["survive", "build"],
                unacceptable_actions=["explore", "mine_ore", "tunnel"],
                expected_risk_level="high",
                tags=["night", "no_shelter"],
            ),
            Scenario(
                id="surv_05", name="夜间有庇护所",
                description="夜晚，已有庇护所，安全等待",
                category="survival", difficulty="easy",
                env_state=s({"time": "night", "has_shelter": True}),
                acceptable_actions=["wait", "craft_tool", "craft_item", "sort_inventory", "check_recipe", "smelt"],
                acceptable_modes=["survive", "craft"],
                unacceptable_actions=["explore", "attack"],
                expected_risk_level="low",
                tags=["night", "has_shelter", "safe"],
            ),
            Scenario(
                id="surv_06", name="溺水中",
                description="在水中，氧气快耗尽，生命值 10",
                category="survival", difficulty="hard",
                env_state=s({
                    "health": 10,
                    "nearby_blocks": ["water"],
                    "breath": 2,
                }),
                acceptable_actions=["swim", "jump", "retreat"],
                acceptable_modes=["survive", "retreat"],
                unacceptable_actions=["dig_down", "mine_stone", "attack"],
                expected_risk_level="high",
                tags=["drowning"],
            ),

            # ═══════════ 战斗 ═══════════
            Scenario(
                id="comb_01", name="单个僵尸近距离",
                description="一只僵尸距离 3 格，有石剑，血量 18",
                category="combat", difficulty="easy",
                env_state=s({
                    "health": 18,
                    "inventory": {"stone_sword": 1},
                    "nearby_entities": [{"type": "zombie", "hostile": True, "distance": 3}],
                }),
                acceptable_actions=["attack", "attack_entity", "equip"],
                acceptable_modes=["combat", "survive"],
                unacceptable_actions=["explore", "mine_stone", "gather_wood"],
                expected_risk_level="medium",
                tags=["hostile", "has_weapon"],
            ),
            Scenario(
                id="comb_02", name="多敌无武器",
                description="3 只僵尸靠近，无武器，血量 15",
                category="combat", difficulty="hard",
                env_state=s({
                    "health": 15,
                    "nearby_entities": [
                        {"type": "zombie", "hostile": True, "distance": 4},
                        {"type": "zombie", "hostile": True, "distance": 6},
                        {"type": "skeleton", "hostile": True, "distance": 8},
                    ],
                }),
                acceptable_actions=["retreat", "flee_from", "sprint"],
                acceptable_modes=["retreat", "survive"],
                unacceptable_actions=["attack", "explore", "mine_stone", "gather_wood"],
                expected_risk_level="high",
                tags=["hostile", "no_weapon", "outnumbered"],
            ),
            Scenario(
                id="comb_03", name="低血量遭遇战",
                description="血量 5，一只骷髅距离 6，有石剑",
                category="combat", difficulty="hard",
                env_state=s({
                    "health": 5,
                    "inventory": {"stone_sword": 1, "apple": 1},
                    "nearby_entities": [{"type": "skeleton", "hostile": True, "distance": 6}],
                }),
                acceptable_actions=["eat_food", "retreat", "flee_from"],
                acceptable_modes=["survive", "retreat"],
                unacceptable_actions=["attack", "explore", "mine_stone"],
                expected_risk_level="high",
                tags=["critical_health", "hostile"],
            ),
            Scenario(
                id="comb_04", name="满血有装备近战",
                description="满血，石剑+铁甲，一只蜘蛛距离 4",
                category="combat", difficulty="easy",
                env_state=s({
                    "inventory": {"stone_sword": 1, "iron_chestplate": 1},
                    "nearby_entities": [{"type": "spider", "hostile": True, "distance": 4}],
                }),
                acceptable_actions=["attack", "attack_entity", "equip"],
                acceptable_modes=["combat", "survive"],
                expected_risk_level="medium",
                tags=["hostile", "well_equipped"],
            ),

            # ═══════════ 资源采集 ═══════════
            Scenario(
                id="res_01", name="开局空手",
                description="游戏开始，空库存，白天，附近有树和石头",
                category="resource", difficulty="easy",
                env_state=s({}),
                acceptable_actions=["gather_wood", "find_resource", "explore", "dig", "punch_node"],
                acceptable_modes=["gather", "explore"],
                unacceptable_actions=["attack", "build_shelter", "smelt"],
                expected_risk_level="low",
                tags=["early_game", "no_tools"],
            ),
            Scenario(
                id="res_02", name="有木无工具",
                description="有 10 木头，无工具，附近有石头",
                category="resource", difficulty="easy",
                env_state=s({
                    "inventory": {"wood": 10, "stick": 4},
                    "nearby_blocks": ["stone", "tree", "dirt"],
                }),
                acceptable_actions=["craft_tool", "craft_item"],
                acceptable_modes=["craft", "gather"],
                unacceptable_actions=["attack", "smelt"],
                expected_risk_level="low",
                tags=["has_wood", "no_tools", "crafting_ready"],
            ),
            Scenario(
                id="res_03", name="石制工具升级",
                description="有木镐，附近有石头，该升级石镐",
                category="resource", difficulty="easy",
                env_state=s({
                    "inventory": {"wooden_pickaxe": 1, "wood": 5, "stick": 4},
                    "nearby_blocks": ["stone", "tree"],
                }),
                acceptable_actions=["mine_stone", "dig", "craft_tool"],
                acceptable_modes=["gather", "craft"],
                expected_risk_level="low",
                tags=["tool_upgrade"],
            ),
            Scenario(
                id="res_04", name="矿石采集",
                description="有石镐，发现铁矿脉",
                category="resource", difficulty="medium",
                env_state=s({
                    "inventory": {"stone_pickaxe": 1},
                    "nearby_blocks": ["iron_ore", "stone", "coal"],
                }),
                acceptable_actions=["mine_ore", "dig", "equip"],
                acceptable_modes=["gather"],
                expected_risk_level="low",
                tags=["mining", "has_tools"],
            ),
            Scenario(
                id="res_05", name="食物采集",
                description="饥饿值 8，附近有被动生物和农作物",
                category="resource", difficulty="easy",
                env_state=s({
                    "hunger": 8,
                    "nearby_entities": [{"type": "sheep", "hostile": False, "distance": 6}],
                    "nearby_blocks": ["farming:wheat_8", "tree"],
                }),
                acceptable_actions=["farm_harvest", "attack_entity", "find_resource", "eat_food"],
                acceptable_modes=["gather", "survive"],
                expected_risk_level="low",
                tags=["low_hunger", "food_available"],
            ),
            Scenario(
                id="res_06", name="库存已满",
                description="库存几乎满了，附近有箱子",
                category="resource", difficulty="easy",
                env_state=s({
                    "inventory": {"wood": 64, "stone": 64, "dirt": 64, "cobble": 64,
                                  "iron_lump": 10, "coal": 20},
                    "nearby_blocks": ["default:chest"],
                }),
                acceptable_actions=["deposit_item", "drop_item", "sort_inventory"],
                acceptable_modes=["gather", "build"],
                expected_risk_level="low",
                tags=["inventory_full"],
            ),

            # ═══════════ 合成/冶炼 ═══════════
            Scenario(
                id="craft_01", name="基础工具合成",
                description="有木头和木棍，需要制作木镐",
                category="craft", difficulty="easy",
                env_state=s({"inventory": {"wood": 8, "stick": 6}}),
                acceptable_actions=["craft_tool", "craft_item"],
                acceptable_modes=["craft"],
                expected_risk_level="low",
                tags=["crafting", "early_game"],
            ),
            Scenario(
                id="craft_02", name="冶炼铁锭",
                description="有铁矿、煤炭，有熔炉",
                category="craft", difficulty="medium",
                env_state=s({
                    "inventory": {"iron_lump": 5, "coal_lump": 3},
                    "nearby_blocks": ["default:furnace"],
                }),
                acceptable_actions=["smelt", "use_node"],
                acceptable_modes=["craft"],
                expected_risk_level="low",
                tags=["smelting"],
            ),
            Scenario(
                id="craft_03", name="查询配方",
                description="想做石剑但不确定材料",
                category="craft", difficulty="easy",
                env_state=s({"inventory": {"cobble": 10, "stick": 4}}),
                acceptable_actions=["check_recipe", "craft_tool"],
                acceptable_modes=["craft"],
                expected_risk_level="low",
                tags=["recipe_check"],
            ),

            # ═══════════ 建造 ═══════════
            Scenario(
                id="build_01", name="紧急庇护所",
                description="天快黑了，有足够材料建庇护所",
                category="survival", difficulty="medium",
                env_state=s({
                    "time": "day",  # 即将入夜
                    "inventory": {"cobble": 30, "wood": 10},
                    "nearby_blocks": ["dirt", "stone"],
                }),
                acceptable_actions=["build_shelter", "place_block", "place_at", "light_area"],
                acceptable_modes=["build", "survive"],
                expected_risk_level="medium",
                tags=["building", "pre_night"],
            ),
            Scenario(
                id="build_02", name="搭桥过沟",
                description="前方是深沟，需要搭桥",
                category="explore", difficulty="medium",
                env_state=s({
                    "inventory": {"cobble": 20},
                    "nearby_blocks": ["air", "stone"],  # 空气=悬崖
                }),
                acceptable_actions=["bridge", "place_block", "place_at"],
                acceptable_modes=["explore", "build"],
                expected_risk_level="medium",
                tags=["bridging", "obstacle"],
            ),
            Scenario(
                id="build_03", name="照明黑暗区域",
                description="在矿洞中，光线很暗，有火把",
                category="explore", difficulty="easy",
                env_state=s({
                    "inventory": {"torch": 5, "stone_pickaxe": 1},
                    "nearby_blocks": ["stone", "coal"],
                    "light_level": 2,
                }),
                acceptable_actions=["light_area", "place_block"],
                acceptable_modes=["explore", "build"],
                expected_risk_level="medium",
                tags=["lighting", "underground"],
            ),

            # ═══════════ 探索 ═══════════
            Scenario(
                id="expl_01", name="安全日间探索",
                description="白天，满血满饱，有基础工具，无敌人",
                category="explore", difficulty="easy",
                env_state=s({
                    "inventory": {"wooden_pickaxe": 1, "wood": 5, "apple": 2},
                }),
                acceptable_actions=["explore", "find_resource", "look_around", "move_to"],
                acceptable_modes=["explore", "gather"],
                expected_risk_level="low",
                tags=["safe_explore"],
            ),
            Scenario(
                id="expl_02", name="矿洞深处探索",
                description="在地下 y=-30，有石镐和火把",
                category="explore", difficulty="medium",
                env_state=s({
                    "position": {"x": 100, "y": -30, "z": 100},
                    "inventory": {"stone_pickaxe": 1, "torch": 3, "apple": 1},
                    "nearby_blocks": ["stone", "iron_ore", "coal"],
                    "light_level": 3,
                }),
                acceptable_actions=["mine_ore", "dig", "tunnel", "light_area", "explore"],
                acceptable_modes=["gather", "explore"],
                expected_risk_level="medium",
                tags=["underground", "mining"],
            ),
            Scenario(
                id="expl_03", name="位置卡住",
                description="已连续 8 步未移动，需要脱困",
                category="explore", difficulty="medium",
                env_state=s({
                    "inventory": {"wooden_pickaxe": 1},
                    "nearby_blocks": ["stone", "dirt"],
                }),
                acceptable_actions=["explore", "move_to", "jump", "dig", "dig_down", "dig_up", "tunnel"],
                acceptable_modes=["explore", "gather"],
                expected_risk_level="low",
                tags=["stuck"],
            ),

            # ═══════════ 复合情境 ═══════════
            Scenario(
                id="comp_01", name="夜间低血量有敌人",
                description="夜晚，血量 6，2 只僵尸，无庇护所，有石剑",
                category="survival", difficulty="hard",
                env_state=s({
                    "time": "night",
                    "health": 6,
                    "inventory": {"stone_sword": 1, "cobble": 10, "apple": 1},
                    "nearby_entities": [
                        {"type": "zombie", "hostile": True, "distance": 5},
                        {"type": "zombie", "hostile": True, "distance": 8},
                    ],
                }),
                acceptable_actions=["eat_food", "retreat", "flee_from", "build_shelter", "sprint"],
                acceptable_modes=["survive", "retreat"],
                unacceptable_actions=["explore", "mine_stone", "gather_wood"],
                expected_risk_level="high",
                tags=["night", "critical_health", "hostile", "compound_danger"],
            ),
            Scenario(
                id="comp_02", name="饥饿死亡边缘有工具",
                description="饥饿值 1，有石剑，附近有鸡",
                category="survival", difficulty="hard",
                env_state=s({
                    "hunger": 1,
                    "inventory": {"stone_sword": 1},
                    "nearby_entities": [{"type": "chicken", "hostile": False, "distance": 4}],
                }),
                acceptable_actions=["attack_entity", "attack", "find_resource"],
                acceptable_modes=["survive", "gather"],
                unacceptable_actions=["explore", "mine_stone", "build_shelter"],
                expected_risk_level="high",
                tags=["critical_hunger", "food_nearby"],
            ),
            Scenario(
                id="comp_03", name="安全基地内整理",
                description="夜晚，有庇护所，满血，库存丰富",
                category="craft", difficulty="easy",
                env_state=s({
                    "time": "night",
                    "has_shelter": True,
                    "inventory": {"wood": 20, "cobble": 30, "iron_lump": 5,
                                  "coal_lump": 3, "stick": 10, "apple": 3},
                    "nearby_blocks": ["default:furnace", "default:chest"],
                }),
                acceptable_actions=["craft_tool", "craft_item", "smelt", "sort_inventory",
                                    "deposit_item", "check_recipe", "wait"],
                acceptable_modes=["craft", "survive"],
                expected_risk_level="low",
                tags=["safe", "crafting_opportunity"],
            ),
            Scenario(
                id="comp_04", name="僵尸群夜间围攻",
                description="夜晚，5只僵尸，有庇护所但被包围",
                category="combat", difficulty="hard",
                env_state=s({
                    "time": "night",
                    "health": 14,
                    "has_shelter": True,
                    "inventory": {"stone_sword": 1, "cobble": 20, "apple": 2},
                    "nearby_entities": [
                        {"type": "zombie", "hostile": True, "distance": 3},
                        {"type": "zombie", "hostile": True, "distance": 4},
                        {"type": "zombie", "hostile": True, "distance": 5},
                        {"type": "zombie", "hostile": True, "distance": 6},
                        {"type": "zombie", "hostile": True, "distance": 7},
                    ],
                }),
                acceptable_actions=["attack", "attack_entity", "equip", "wait",
                                    "retreat", "place_block"],
                acceptable_modes=["combat", "survive", "retreat"],
                unacceptable_actions=["explore", "mine_stone", "gather_wood"],
                expected_risk_level="high",
                tags=["night", "hostile", "outnumbered", "has_shelter"],
            ),

            # ═══════════ 农业 ═══════════
            Scenario(
                id="farm_01", name="种植小麦",
                description="有种子和锄头，附近有耕地",
                category="resource", difficulty="easy",
                env_state=s({
                    "inventory": {"wheat_seed": 10, "wooden_hoe": 1},
                    "nearby_blocks": ["farming:soil_wet", "water"],
                }),
                acceptable_actions=["farm_plant", "equip"],
                acceptable_modes=["gather", "build"],
                expected_risk_level="low",
                tags=["farming", "planting"],
            ),
            Scenario(
                id="farm_02", name="收获农作物",
                description="成熟的小麦田，该收获了",
                category="resource", difficulty="easy",
                env_state=s({
                    "nearby_blocks": ["farming:wheat_8", "farming:wheat_8", "water"],
                }),
                acceptable_actions=["farm_harvest", "dig"],
                acceptable_modes=["gather"],
                expected_risk_level="low",
                tags=["farming", "harvesting"],
            ),

            # ═══════════ 容器交互 ═══════════
            Scenario(
                id="cont_01", name="从箱子取工具",
                description="附近有箱子，需要取出铁镐",
                category="resource", difficulty="easy",
                env_state=s({
                    "nearby_blocks": ["default:chest"],
                    "inventory": {"wood": 5},
                }),
                acceptable_actions=["take_from_container", "use_node"],
                acceptable_modes=["gather", "craft"],
                expected_risk_level="low",
                tags=["container"],
            ),

            # ═══════════ 边界/陷阱场景 ═══════════
            Scenario(
                id="edge_01", name="完全空状态",
                description="最小状态：空库存、无实体、白天",
                category="explore", difficulty="easy",
                env_state=s({}),
                acceptable_actions=["explore", "gather_wood", "find_resource", "look_around", "punch_node"],
                acceptable_modes=["gather", "explore"],
                expected_risk_level="low",
                tags=["edge_case", "empty_state"],
            ),
            Scenario(
                id="edge_02", name="满血满饱全装备",
                description="最佳状态：满属性、铁制全套、有庇护所",
                category="explore", difficulty="easy",
                env_state=s({
                    "has_shelter": True,
                    "inventory": {
                        "iron_pickaxe": 1, "iron_sword": 1, "iron_axe": 1,
                        "apple": 10, "cobble": 64, "wood": 30, "torch": 10,
                        "iron_chestplate": 1,
                    },
                }),
                acceptable_actions=["explore", "mine_ore", "find_resource", "tunnel",
                                    "move_to", "look_around"],
                acceptable_modes=["explore", "gather"],
                expected_risk_level="low",
                tags=["edge_case", "optimal_state"],
            ),
            Scenario(
                id="edge_03", name="血量为 1",
                description="几乎死亡，只剩 1 血，有食物",
                category="survival", difficulty="hard",
                env_state=s({"health": 1, "inventory": {"apple": 5}}),
                acceptable_actions=["eat_food"],
                acceptable_modes=["survive"],
                unacceptable_actions=["explore", "attack", "dig", "mine_stone",
                                      "mine_ore", "tunnel", "gather_wood"],
                expected_risk_level="high",
                tags=["edge_case", "near_death"],
            ),

            # ═══════════ 更多生存细分 ═══════════
            Scenario(
                id="surv_07", name="低血量无食物有动物",
                description="血量 4，无食物，附近有猪",
                category="survival", difficulty="hard",
                env_state=s({
                    "health": 4,
                    "hunger": 10,
                    "nearby_entities": [{"type": "pig", "hostile": False, "distance": 5}],
                }),
                acceptable_actions=["attack_entity", "attack", "find_resource"],
                acceptable_modes=["survive", "gather"],
                unacceptable_actions=["explore", "mine_stone", "tunnel"],
                expected_risk_level="high",
                tags=["critical_health", "food_nearby"],
            ),
            Scenario(
                id="surv_08", name="夜间低饥饿有庇护所",
                description="夜晚，有庇护所，饥饿值 4，有小麦",
                category="survival", difficulty="medium",
                env_state=s({
                    "time": "night",
                    "has_shelter": True,
                    "hunger": 4,
                    "inventory": {"bread": 2},
                }),
                acceptable_actions=["eat_food", "wait", "craft_item"],
                acceptable_modes=["survive"],
                expected_risk_level="medium",
                tags=["night", "low_hunger", "has_food", "has_shelter"],
            ),
            Scenario(
                id="surv_09", name="氧气耗尽水下",
                description="水下，氧气 0，血量 12",
                category="survival", difficulty="hard",
                env_state=s({
                    "health": 12,
                    "breath": 0,
                    "nearby_blocks": ["water"],
                }),
                acceptable_actions=["swim", "jump"],
                acceptable_modes=["survive", "retreat"],
                unacceptable_actions=["dig_down", "mine_stone", "explore", "attack"],
                expected_risk_level="high",
                tags=["drowning", "critical"],
            ),
            Scenario(
                id="surv_10", name="夜间被骷髅远程攻击",
                description="夜晚，被骷髅远程射击，血量 10，无庇护所",
                category="survival", difficulty="hard",
                env_state=s({
                    "time": "night",
                    "health": 10,
                    "nearby_entities": [{"type": "skeleton", "hostile": True, "distance": 10}],
                    "inventory": {"cobble": 15, "wood": 5},
                }),
                acceptable_actions=["build_shelter", "retreat", "flee_from", "sprint",
                                    "place_block"],
                acceptable_modes=["survive", "retreat", "build"],
                unacceptable_actions=["explore", "mine_stone", "gather_wood"],
                expected_risk_level="high",
                tags=["night", "hostile", "ranged_attack"],
            ),

            # ═══════════ 更多资源/合成 ═══════════
            Scenario(
                id="res_07", name="发现钻石矿",
                description="深层发现钻石矿，有铁镐",
                category="resource", difficulty="medium",
                env_state=s({
                    "position": {"x": 100, "y": -50, "z": 100},
                    "inventory": {"iron_pickaxe": 1, "torch": 5},
                    "nearby_blocks": ["diamond_ore", "stone", "lava"],
                }),
                acceptable_actions=["mine_ore", "dig", "equip", "light_area"],
                acceptable_modes=["gather"],
                unacceptable_actions=["explore", "build_shelter"],
                expected_risk_level="medium",
                tags=["mining", "diamond", "underground"],
            ),
            Scenario(
                id="res_08", name="食物短缺寻找水源",
                description="饥饿值 6，附近无动物无农田，需探索",
                category="resource", difficulty="medium",
                env_state=s({
                    "hunger": 6,
                    "nearby_blocks": ["dirt", "stone"],
                }),
                acceptable_actions=["find_resource", "explore", "look_around"],
                acceptable_modes=["gather", "explore", "survive"],
                expected_risk_level="medium",
                tags=["low_hunger", "no_food_nearby"],
            ),
            Scenario(
                id="craft_04", name="铁工具全套合成",
                description="有铁锭、煤、木棍，可以合成铁制工具",
                category="craft", difficulty="medium",
                env_state=s({
                    "inventory": {"iron_ingot": 10, "stick": 8, "coal_lump": 5},
                }),
                acceptable_actions=["craft_tool", "craft_item"],
                acceptable_modes=["craft"],
                expected_risk_level="low",
                tags=["crafting", "iron_tools"],
            ),

            # ═══════════ 更多复合 ═══════════
            Scenario(
                id="comp_05", name="日出后安全出发",
                description="刚天亮，有庇护所和基础装备，该出发探索",
                category="explore", difficulty="easy",
                env_state=s({
                    "has_shelter": True,
                    "inventory": {"stone_pickaxe": 1, "stone_sword": 1,
                                  "apple": 3, "torch": 5, "wood": 10},
                }),
                acceptable_actions=["explore", "find_resource", "move_to", "look_around"],
                acceptable_modes=["explore", "gather"],
                expected_risk_level="low",
                tags=["day", "well_equipped", "safe"],
            ),
            Scenario(
                id="comp_06", name="黄昏赶回基地",
                description="即将入夜，距基地较远，需要快速返回",
                category="survival", difficulty="medium",
                env_state=s({
                    "position": {"x": 300, "y": 10, "z": 300},
                    "inventory": {"stone_sword": 1, "torch": 2, "cobble": 10},
                }),
                acceptable_actions=["move_to", "sprint", "retreat", "build_shelter"],
                acceptable_modes=["survive", "explore"],
                expected_risk_level="medium",
                tags=["pre_night", "far_from_base"],
            ),
            Scenario(
                id="comp_07", name="掉落悬崖",
                description="从高处掉落，血量 8，在峡谷底部",
                category="survival", difficulty="medium",
                env_state=s({
                    "health": 8,
                    "position": {"x": 100, "y": -10, "z": 100},
                    "nearby_blocks": ["stone", "dirt"],
                    "inventory": {"wooden_pickaxe": 1, "apple": 1},
                }),
                acceptable_actions=["eat_food", "tower_up", "dig_up", "jump",
                                    "explore", "find_resource"],
                acceptable_modes=["survive", "explore"],
                expected_risk_level="medium",
                tags=["low_health", "trapped"],
            ),
            Scenario(
                id="comp_08", name="岩浆附近采矿",
                description="采矿时发现岩浆，需要小心",
                category="resource", difficulty="hard",
                env_state=s({
                    "position": {"x": 100, "y": -40, "z": 100},
                    "inventory": {"iron_pickaxe": 1, "cobble": 20, "torch": 3},
                    "nearby_blocks": ["lava", "stone", "iron_ore"],
                }),
                acceptable_actions=["place_block", "retreat", "mine_ore", "dig"],
                acceptable_modes=["gather", "survive"],
                unacceptable_actions=["dig_down", "jump"],
                expected_risk_level="high",
                tags=["lava", "mining", "danger"],
            ),
        ]

        self.scenarios = scenarios
