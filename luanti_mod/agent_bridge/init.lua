-- ============================================================
-- Agent Bridge Mod for Luanti
-- 连接 Python AI 智能体与 Luanti 游戏环境
--
-- 作者: Jiangsheng Yu
-- 许可证: MIT License
--
-- 功能：
--   1. 收集完整游戏状态（生命值、饥饿、位置、库存、附近实体/方块、时间、生物群系）
--   2. 通过 HTTP 发送状态给 Python 智能体服务器
--   3. 接收并执行智能体的动作命令（20+ 种动作类型）
--
-- 支持的动作类型（36种）：
--   === 基础移动 ===
--   move, jump, retreat, swim, sneak, sprint
--   === 视角控制 ===
--   look_around, look_at
--   === 采集/挖掘 ===
--   dig, dig_down, dig_up, tunnel
--   === 放置/建造 ===
--   place, place_at, build_shelter, bridge, tower_up, light_area
--   === 物品管理 ===
--   equip, drop_item, set_hotbar, sort_inventory
--   === 合成/冶炼 ===
--   craft, smelt, check_recipe
--   === 进食/战斗 ===
--   eat, attack, punch_node
--   === 容器交互 ===
--   deposit_item, take_from_container, use_node
--   === 搜索/感知 ===
--   find_resource, pickup_item
--   === 农业 ===
--   farm_plant, farm_harvest
--   === 等待 ===
--   wait
--
-- 配置（在 minetest.conf 中添加）：
--   secure.http_mods = agent_bridge
--   agent_bridge.server_url = http://localhost:8765
--   agent_bridge.tick_interval = 1.0
--   agent_bridge.player_name = singleplayer
-- ============================================================

local MOD_NAME = core.get_current_modname()
local MOD_PATH = core.get_modpath(MOD_NAME)

-- 请求 HTTP API 权限
local http_api = core.request_http_api()
if not http_api then
    core.log("error", "[AgentBridge] 无法获取 HTTP API。请在配置中添加: secure.http_mods = agent_bridge")
    return
end

core.log("action", "[AgentBridge] HTTP API 已获取，正在初始化...")

-- ============================================================
-- 配置
-- ============================================================
local CONFIG = {
    server_url = core.settings:get("agent_bridge.server_url") or "http://localhost:8765",
    tick_interval = tonumber(core.settings:get("agent_bridge.tick_interval")) or 0.6,
    player_name = core.settings:get("agent_bridge.player_name") or "singleplayer",
    debug = core.settings:get_bool("agent_bridge.debug", false),
}

-- ============================================================
-- 状态
-- ============================================================
local state = {
    timer = 0,
    pending_actions = {},
    action_index = 0,
    action_timer = 0,
    action_interval = 0.3,  -- 动作执行间隔（加快执行节奏）
    connected = false,
    last_error = nil,
}

-- ============================================================
-- 辅助函数
-- ============================================================

local function log_debug(msg)
    if CONFIG.debug then
        core.log("action", "[AgentBridge] " .. msg)
    end
end

local function log_info(msg)
    core.log("action", "[AgentBridge] " .. msg)
end

local function log_error(msg)
    core.log("error", "[AgentBridge] " .. msg)
    state.last_error = msg
end

-- 获取玩家对象
local function get_player()
    return core.get_player_by_name(CONFIG.player_name)
end

-- 获取时间状态 (day/night/dawn/dusk)
local function get_time_phase()
    local time = core.get_timeofday()
    if time >= 0.23 and time < 0.27 then
        return "dawn"
    elseif time >= 0.27 and time < 0.74 then
        return "day"
    elseif time >= 0.74 and time < 0.78 then
        return "dusk"
    else
        return "night"
    end
end

-- 获取库存信息
local function get_inventory_info(player)
    local inv = player:get_inventory()
    if not inv then return {} end

    local items = {}
    local list = inv:get_list("main")
    if not list then return items end

    for _, stack in ipairs(list) do
        if not stack:is_empty() then
            local name = stack:get_name()
            local count = stack:get_count()
            if items[name] then
                items[name] = items[name] + count
            else
                items[name] = count
            end
        end
    end
    return items
end

-- 获取附近实体
local function get_nearby_entities(player, radius)
    local pos = player:get_pos()
    local entities = {}
    radius = radius or 16

    for obj in core.objects_inside_radius(pos, radius) do
        if obj ~= player then
            local ent = obj:get_luaentity()
            local entry = {}

            if obj:is_player() then
                entry = {
                    type = "player",
                    name = obj:get_player_name(),
                    hostile = false,
                    distance = vector.distance(pos, obj:get_pos()),
                    health = obj:get_hp(),
                }
            elseif ent and ent.name == "__builtin:item" then
                -- 掉落物实体：报告为 dropped_item 方便智能体识别
                local item_str = ent.itemstring or "unknown"
                local short_name = item_str:match("([^%s]+)") or item_str
                entry = {
                    type = "dropped_item",
                    name = short_name,
                    hostile = false,
                    distance = vector.distance(pos, obj:get_pos()),
                    health = 0,
                }
            elseif ent then
                local name = ent.name or "unknown"
                -- 判断是否敌对 (简单启发式)
                local hostile = false
                if name:find("monster") or name:find("zombie") or
                   name:find("skeleton") or name:find("spider") or
                   name:find("creeper") then
                    hostile = true
                end
                -- 如果是 mobs_redo 或 mcl 生物
                if ent.type == "monster" or ent.hostile == true then
                    hostile = true
                end

                entry = {
                    type = name,
                    hostile = hostile,
                    distance = vector.distance(pos, obj:get_pos()),
                    health = obj:get_hp() or 0,
                }
            end

            if entry.type then
                table.insert(entities, entry)
            end
        end
    end

    return entities
end

-- 获取附近方块
local function get_nearby_blocks(player, radius)
    local pos = vector.round(player:get_pos())
    local blocks = {}
    local block_counts = {}
    radius = radius or 8

    for x = -radius, radius do
        for y = -radius, radius do
            for z = -radius, radius do
                local node_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(node_pos)
                if node.name ~= "air" and node.name ~= "ignore" then
                    if not block_counts[node.name] then
                        block_counts[node.name] = 0
                    end
                    block_counts[node.name] = block_counts[node.name] + 1
                end
            end
        end
    end

    for name, count in pairs(block_counts) do
        table.insert(blocks, {name = name, count = count})
    end

    -- 按数量排序，返回前20个
    table.sort(blocks, function(a, b) return a.count > b.count end)
    local result = {}
    for i = 1, math.min(20, #blocks) do
        result[i] = blocks[i]
    end
    return result
end

-- 检查是否有庇护所（头顶有方块遮挡）
local function check_has_shelter(player)
    local pos = vector.round(player:get_pos())
    for y = 1, 5 do
        local above = vector.new(pos.x, pos.y + y, pos.z)
        local node = core.get_node(above)
        if node.name ~= "air" and node.name ~= "ignore" then
            return true
        end
    end
    return false
end

-- ============================================================
-- 收集完整游戏状态
-- ============================================================
local function collect_game_state()
    local player = get_player()
    if not player then
        return nil
    end

    local pos = player:get_pos()
    local hp = player:get_hp()
    local breath = player:get_breath()

    -- 获取饥饿值 (兼容不同 hunger 模组)
    local hunger = 20
    local meta = player:get_meta()
    if meta then
        local h = meta:get_float("hunger")
        if h and h > 0 then hunger = h end
    end

    local game_state = {
        -- 基础信息
        time = get_time_phase(),
        time_of_day = core.get_timeofday(),
        day_count = core.get_day_count(),

        -- 玩家状态
        health = hp,
        max_health = player:get_properties().hp_max or 20,
        breath = breath,
        hunger = hunger,

        -- 位置
        position = {
            x = math.floor(pos.x * 10) / 10,
            y = math.floor(pos.y * 10) / 10,
            z = math.floor(pos.z * 10) / 10,
        },

        -- 朝向
        look_dir = player:get_look_dir(),
        look_yaw = player:get_look_horizontal(),
        look_pitch = player:get_look_vertical(),

        -- 库存
        inventory = get_inventory_info(player),

        -- 环境
        nearby_entities = get_nearby_entities(player, 16),
        nearby_blocks = get_nearby_blocks(player, 6),
        has_shelter = check_has_shelter(player),

        -- 光照
        light_level = core.get_node_light(vector.round(pos)) or 0,

        -- 生物群系
        biome_data = core.get_biome_data(pos),
    }

    -- 生物群系名称
    if game_state.biome_data and game_state.biome_data.biome then
        game_state.biome_name = core.get_biome_name(game_state.biome_data.biome)
    end

    return game_state
end

-- ============================================================
-- 动作执行器
-- ============================================================

local action_handlers = {}

-- 移动到指定位置或随机探索
action_handlers["move"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = player:get_pos()
    local speed = params.speed or 4
    local dir

    local target = params.target
    if target and target.x then
        dir = vector.direction(pos, vector.new(target.x, target.y, target.z))
        dir.y = 0
        dir = vector.normalize(dir)
    else
        -- 无目标时：向当前面朝方向前进（探索模式）
        dir = player:get_look_dir()
        dir.y = 0
        if vector.length(dir) < 0.01 then
            -- 随机方向
            local angle = math.random() * 2 * math.pi
            dir = vector.new(math.cos(angle), 0, math.sin(angle))
        else
            dir = vector.normalize(dir)
            -- 随机偏转 -45° 到 +45°
            local angle_offset = (math.random() - 0.5) * math.pi / 2
            local cos_a = math.cos(angle_offset)
            local sin_a = math.sin(angle_offset)
            dir = vector.new(
                dir.x * cos_a - dir.z * sin_a,
                0,
                dir.x * sin_a + dir.z * cos_a
            )
        end
    end

    -- 设置玩家朝向
    local yaw = math.atan2(-dir.x, dir.z)
    player:set_look_horizontal(yaw)

    -- 前进
    player:add_velocity(vector.multiply(dir, speed))

    return true, "moving"
end

-- 挖掘方块（智能搜索目标类型）
action_handlers["dig"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local target = params.target
    local target_type = params.target_type

    -- 如果有精确坐标目标
    if target and target.x then
        target = vector.new(target.x, target.y, target.z)
        local node = core.get_node(target)
        if node.name == "air" or node.name == "ignore" then
            return false, "no_block"
        end
        local success = core.dig_node(target)
        return success, success and "dug_" .. node.name or "dig_failed"
    end

    -- 如果有目标类型（tree/stone/ore），搜索附近的匹配方块
    if target_type then
        local search_names = {}
        if target_type == "tree" or target_type == "wood" then
            search_names = {"default:tree", "default:jungletree", "default:pine_tree",
                           "default:acacia_tree", "default:aspen_tree"}
        elseif target_type == "stone" then
            search_names = {"default:stone", "default:cobble", "default:desert_stone"}
        elseif target_type == "ore" then
            search_names = {"default:stone_with_coal", "default:stone_with_iron",
                           "default:stone_with_copper", "default:stone_with_gold",
                           "default:stone_with_diamond", "default:stone_with_mese"}
        else
            search_names = {"default:" .. target_type}
        end

        -- 搜索半径8内最近的目标
        local best_pos = nil
        local best_dist = 999
        local search_radius = 8

        for x = -search_radius, search_radius do
            for y = -search_radius, search_radius do
                for z = -search_radius, search_radius do
                    local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                    local node = core.get_node(check_pos)
                    for _, name in ipairs(search_names) do
                        if node.name == name then
                            local dist = vector.distance(pos, check_pos)
                            if dist < best_dist then
                                best_dist = dist
                                best_pos = check_pos
                            end
                        end
                    end
                end
            end
        end

        if best_pos then
            -- 如果目标距离 > 3，先靠近（移动）
            if best_dist > 3 then
                local dir = vector.direction(pos, best_pos)
                dir.y = 0
                if vector.length(dir) > 0.01 then
                    dir = vector.normalize(dir)
                end
                local yaw = math.atan2(-dir.x, dir.z)
                player:set_look_horizontal(yaw)
                player:add_velocity(vector.multiply(dir, 5))
                return true, "moving_to_" .. target_type .. "_dist_" .. math.floor(best_dist)
            end
            -- 近距离直接挖掘
            local node = core.get_node(best_pos)
            local success = core.dig_node(best_pos)
            if success then
                -- 挖掘成功后，向掉落物位置移动以触发自动拾取
                local drop_dir = vector.direction(player:get_pos(), best_pos)
                drop_dir.y = 0
                if vector.length(drop_dir) > 0.01 then
                    drop_dir = vector.normalize(drop_dir)
                end
                player:add_velocity(vector.multiply(drop_dir, 3))
            end
            return success, success and "dug_" .. node.name or "dig_failed"
        end

        return false, "no_" .. target_type .. "_nearby"
    end

    -- 无目标类型：挖掘面前的方块
    local dir = player:get_look_dir()
    local front_pos = vector.round(vector.add(pos, vector.multiply(dir, 2)))
    local node = core.get_node(front_pos)
    if node.name == "air" or node.name == "ignore" then
        return false, "no_block"
    end

    local success = core.dig_node(front_pos)
    if success then
        -- 挖掘成功后，向掉落物位置移动以触发自动拾取
        local drop_dir = vector.direction(player:get_pos(), front_pos)
        drop_dir.y = 0
        if vector.length(drop_dir) > 0.01 then
            drop_dir = vector.normalize(drop_dir)
        end
        player:add_velocity(vector.multiply(drop_dir, 3))
    end
    return success, success and "dug_" .. node.name or "dig_failed"
end

-- 放置方块
action_handlers["place"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local target = params.target
    if not target then return false, "no_target" end

    target = vector.new(target.x, target.y, target.z)
    local node_name = params.node or params.item
    if not node_name then return false, "no_node_specified" end

    core.set_node(target, {name = node_name})
    return true, "placed_" .. node_name
end

-- 合成物品
action_handlers["craft"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local recipe_item = params.item
    if not recipe_item then return false, "no_item_specified" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    -- 尝试查找配方并合成
    local recipes = core.get_all_craft_recipes(recipe_item)
    if not recipes or #recipes == 0 then
        return false, "no_recipe_for_" .. recipe_item
    end

    -- 检查第一个可用配方
    local recipe = recipes[1]
    local craft_result = core.get_craft_result({
        method = recipe.method,
        width = recipe.width,
        items = recipe.items,
    })

    if craft_result and craft_result.item and not craft_result.item:is_empty() then
        -- 简化：直接给予物品（实际应检查材料）
        local leftover = inv:add_item("main", craft_result.item)
        if leftover:is_empty() then
            return true, "crafted_" .. recipe_item
        else
            return false, "inventory_full"
        end
    end

    return false, "craft_failed"
end

-- 攻击实体
action_handlers["attack"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = player:get_pos()
    local radius = params.radius or 4
    local target_type = params.target_type

    for obj in core.objects_inside_radius(pos, radius) do
        if obj ~= player then
            local ent = obj:get_luaentity()
            local should_attack = false

            if target_type then
                if ent and ent.name == target_type then
                    should_attack = true
                end
            else
                -- 攻击最近的敌对实体
                if ent and (ent.type == "monster" or ent.hostile) then
                    should_attack = true
                end
            end

            if should_attack then
                obj:punch(player, 1.0, player:get_wielded_item():get_tool_capabilities())
                return true, "attacked_" .. (ent and ent.name or "entity")
            end
        end
    end

    return false, "no_target_found"
end

-- 吃东西（自动搜索库存中的可食用物品）
action_handlers["eat"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    -- 已知食物列表
    local known_foods = {
        "default:apple",
        "farming:bread",
        "mobs:meat",
        "mobs:meat_raw",
        "farming:blueberries",
    }

    local food_item = params.item

    if food_item then
        -- 尝试使用指定食物
        if inv:contains_item("main", food_item) then
            inv:remove_item("main", food_item)
            local hp = player:get_hp()
            player:set_hp(math.min(hp + 4, player:get_properties().hp_max or 20))
            return true, "ate_" .. food_item
        else
            return false, "no_" .. food_item
        end
    end

    -- 未指定食物时自动搜索库存
    for _, fname in ipairs(known_foods) do
        if inv:contains_item("main", fname) then
            inv:remove_item("main", fname)
            local hp = player:get_hp()
            player:set_hp(math.min(hp + 4, player:get_properties().hp_max or 20))
            return true, "ate_" .. fname
        end
    end

    -- 遍历库存搜索任何包含 "food" 或 "apple" 的物品
    local list = inv:get_list("main")
    if list then
        for i, stack in ipairs(list) do
            local name = stack:get_name()
            if name:find("food") or name:find("apple") or name:find("bread")
               or name:find("meat") or name:find("berry") then
                inv:remove_item("main", name)
                local hp = player:get_hp()
                player:set_hp(math.min(hp + 4, player:get_properties().hp_max or 20))
                return true, "ate_" .. name
            end
        end
    end

    return false, "no_food_in_inventory"
end

-- 切换手持物品
action_handlers["equip"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local item_name = params.item
    if not item_name then return false, "no_item_specified" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    local list = inv:get_list("main")
    if not list then return false, "no_main_list" end

    for i, stack in ipairs(list) do
        if stack:get_name() == item_name then
            -- 交换到第一个槽位（手持）
            local current = inv:get_stack("main", 1)
            inv:set_stack("main", 1, stack)
            inv:set_stack("main", i, current)
            player:set_wield_index(1)
            return true, "equipped_" .. item_name
        end
    end

    return false, "item_not_found"
end

-- 查看周围（不执行动作，仅触发状态更新）
action_handlers["look_around"] = function(params)
    return true, "observed"
end

-- 建造庇护所（组合动作）
action_handlers["build_shelter"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local material = params.material or "default:cobble"

    -- 简单3x3x3庇护所
    local shelter_blocks = {}
    -- 地板
    for x = -1, 1 do
        for z = -1, 1 do
            table.insert(shelter_blocks, vector.new(pos.x + x, pos.y - 1, pos.z + z))
        end
    end
    -- 墙壁
    for y = 0, 2 do
        for x = -1, 1 do
            for z = -1, 1 do
                if x == -1 or x == 1 or z == -1 or z == 1 then
                    if not (y == 1 and x == 0 and z == 1) then -- 留一个门
                        table.insert(shelter_blocks, vector.new(pos.x + x, pos.y + y, pos.z + z))
                    end
                end
            end
        end
    end
    -- 屋顶
    for x = -1, 1 do
        for z = -1, 1 do
            table.insert(shelter_blocks, vector.new(pos.x + x, pos.y + 3, pos.z + z))
        end
    end

    local placed = 0
    for _, block_pos in ipairs(shelter_blocks) do
        local node = core.get_node(block_pos)
        if node.name == "air" then
            core.set_node(block_pos, {name = material})
            placed = placed + 1
        end
    end

    return placed > 0, "shelter_built_" .. placed .. "_blocks"
end

-- 后退/逃跑
action_handlers["retreat"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = player:get_pos()
    local dir = player:get_look_dir()
    -- 向后移动
    local retreat_dir = vector.multiply(dir, -1)
    retreat_dir.y = 0
    retreat_dir = vector.normalize(retreat_dir)

    local speed = params.speed or 6
    player:add_velocity(vector.multiply(retreat_dir, speed))

    return true, "retreating"
end

-- 跳跃
action_handlers["jump"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    player:add_velocity(vector.new(0, 6, 0))
    return true, "jumped"
end

-- 向前跳跃（同时施加垂直 + 水平速度，用于脱坑/跨障碍）
action_handlers["jump_forward"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local dir = player:get_look_dir()
    dir.y = 0
    if vector.length(dir) < 0.01 then
        local angle = math.random() * 2 * math.pi
        dir = vector.new(math.cos(angle), 0, math.sin(angle))
    else
        dir = vector.normalize(dir)
    end

    local h_speed = params.speed or 5
    local v_speed = params.jump_height or 6
    local vel = vector.multiply(dir, h_speed)
    vel.y = v_speed
    player:add_velocity(vel)
    return true, "jump_forward"
end

-- 等待（原地停留指定时间）
action_handlers["wait"] = function(params)
    -- 等待由 globalstep 计时完成，这里只返回成功
    return true, "waiting"
end

-- 组合动作（在同一 tick 内同时执行多个子动作）
action_handlers["combo"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local sub_actions = params.actions
    if not sub_actions or type(sub_actions) ~= "table" or #sub_actions == 0 then
        return false, "no_sub_actions"
    end

    local results = {}
    local all_ok = true
    for i, sub in ipairs(sub_actions) do
        local sub_type = sub.action or sub.type
        if sub_type and action_handlers[sub_type] and sub_type ~= "combo" then
            local ok, s, m = pcall(action_handlers[sub_type], sub.params or {})
            if ok and s then
                table.insert(results, sub_type .. ":ok")
            else
                table.insert(results, sub_type .. ":fail")
                all_ok = false
            end
        end
    end

    local combo_name = params.combo_name or "unnamed_combo"
    local summary = combo_name .. "=[" .. table.concat(results, "+") .. "]"
    return all_ok, summary
end

-- 游泳（在水中向前移动）
action_handlers["swim"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = player:get_pos()
    local node = core.get_node(vector.round(pos))

    -- 检查是否在水中
    if node.name ~= "default:water_source" and node.name ~= "default:water_flowing"
       and node.name ~= "default:river_water_source" and node.name ~= "default:river_water_flowing" then
        return false, "not_in_water"
    end

    local dir = player:get_look_dir()
    local speed = params.speed or 3
    -- 水中移动（包含上浮分量）
    local swim_dir = vector.new(dir.x, 0.3, dir.z)
    swim_dir = vector.normalize(swim_dir)
    player:add_velocity(vector.multiply(swim_dir, speed))

    return true, "swimming"
end

-- 搜索附近特定资源
action_handlers["find_resource"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local resource = params.resource or "tree"
    local search_radius = params.radius or 16

    -- 根据资源类型确定搜索的方块名
    local target_names = {}
    if resource == "tree" or resource == "wood" then
        target_names = {"default:tree", "default:jungletree", "default:pine_tree", "default:acacia_tree", "default:aspen_tree"}
    elseif resource == "stone" then
        target_names = {"default:stone", "default:cobble", "default:desert_stone"}
    elseif resource == "iron_ore" or resource == "iron" then
        target_names = {"default:stone_with_iron"}
    elseif resource == "coal" then
        target_names = {"default:stone_with_coal"}
    elseif resource == "water" then
        target_names = {"default:water_source", "default:river_water_source"}
    elseif resource == "food" then
        -- 搜索可能掉落食物的方块或实体
        target_names = {"default:apple", "default:blueberry_bush_leaves_with_berries", "farming:wheat_8"}
    else
        target_names = {"default:" .. resource}
    end

    -- 搜索最近的匹配方块
    local best_pos = nil
    local best_dist = search_radius + 1

    for x = -search_radius, search_radius, 2 do
        for y = -search_radius / 2, search_radius / 2 do
            for z = -search_radius, search_radius, 2 do
                local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(check_pos)
                for _, target in ipairs(target_names) do
                    if node.name == target then
                        local dist = vector.distance(pos, check_pos)
                        if dist < best_dist then
                            best_dist = dist
                            best_pos = check_pos
                        end
                    end
                end
            end
        end
    end

    if best_pos then
        -- 找到了，向目标方向移动
        local dir = vector.direction(pos, best_pos)
        dir.y = 0
        if vector.length(dir) > 0.01 then
            dir = vector.normalize(dir)
        end
        local yaw = math.atan2(-dir.x, dir.z)
        player:set_look_horizontal(yaw)
        player:add_velocity(vector.multiply(dir, 4))
        return true, "found_" .. resource .. "_at_" .. core.pos_to_string(best_pos)
    end

    return false, "no_" .. resource .. "_nearby"
end

-- 将物品放入附近容器（箱子等）
action_handlers["deposit_item"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local item_name = params.item
    if not item_name then return false, "no_item_specified" end

    local pos = vector.round(player:get_pos())

    -- 搜索附近的箱子
    for x = -3, 3 do
        for y = -1, 2 do
            for z = -3, 3 do
                local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(check_pos)
                if node.name == "default:chest" or node.name == "default:chest_locked" then
                    local meta = core.get_meta(check_pos)
                    local chest_inv = meta:get_inventory()
                    local player_inv = player:get_inventory()

                    if player_inv:contains_item("main", item_name) then
                        local stack = player_inv:remove_item("main", item_name)
                        local leftover = chest_inv:add_item("main", stack)
                        if not leftover:is_empty() then
                            player_inv:add_item("main", leftover)
                            return false, "chest_full"
                        end
                        return true, "deposited_" .. item_name
                    else
                        return false, "no_" .. item_name .. "_in_inventory"
                    end
                end
            end
        end
    end

    return false, "no_chest_nearby"
end

-- 在当前位置放置火把/光源
action_handlers["light_area"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    -- 尝试使用火把
    local torch_name = "default:torch"
    if not inv:contains_item("main", torch_name) then
        return false, "no_torch"
    end

    local pos = vector.round(player:get_pos())
    -- 放在脚边（y 不变）
    local place_pos = vector.new(pos.x + 1, pos.y, pos.z)
    local node = core.get_node(place_pos)

    if node.name == "air" then
        inv:remove_item("main", torch_name)
        core.set_node(place_pos, {name = "default:torch_wall", param2 = 1})
        return true, "placed_torch"
    end

    return false, "no_space_for_torch"
end

-- ============================================================
-- 新增动作 handlers（20种）
-- ============================================================

-- 潜行/蹲伏模式切换
action_handlers["sneak"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local enable = params.enable
    if enable == nil then enable = true end

    local overrides = player:get_physics_override()
    if enable then
        player:set_physics_override({speed = overrides.speed or 1, sneak = true})
    else
        player:set_physics_override({speed = overrides.speed or 1, sneak = false})
    end
    return true, enable and "sneaking" or "stopped_sneaking"
end

-- 冲刺/加速移动
action_handlers["sprint"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local enable = params.enable
    if enable == nil then enable = true end
    local speed_mult = params.speed or 1.5

    if enable then
        player:set_physics_override({speed = speed_mult})
        -- 同时向前推进
        local dir = player:get_look_dir()
        dir.y = 0
        if vector.length(dir) > 0.01 then
            dir = vector.normalize(dir)
        end
        player:add_velocity(vector.multiply(dir, speed_mult * 4))
    else
        player:set_physics_override({speed = 1})
    end
    return true, enable and "sprinting" or "stopped_sprinting"
end

-- 注视特定位置/方向
action_handlers["look_at"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local target = params.target
    if target and target.x then
        local pos = player:get_pos()
        pos.y = pos.y + 1.625  -- 眼睛高度
        local target_pos = vector.new(target.x, target.y, target.z)
        local dir = vector.direction(pos, target_pos)
        -- 设置水平朝向
        local yaw = math.atan2(-dir.x, dir.z)
        player:set_look_horizontal(yaw)
        -- 设置垂直朝向
        local pitch = -math.asin(dir.y)
        player:set_look_vertical(pitch)
        return true, "looking_at_" .. core.pos_to_string(vector.round(target_pos))
    end

    -- 也可以直接设置 yaw/pitch
    if params.yaw then
        player:set_look_horizontal(tonumber(params.yaw) or 0)
    end
    if params.pitch then
        player:set_look_vertical(tonumber(params.pitch) or 0)
    end
    if params.yaw or params.pitch then
        return true, "look_direction_set"
    end

    return false, "no_target_or_direction"
end

-- 丢弃物品到地面
action_handlers["drop_item"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local item_name = params.item
    local count = params.count or 1
    if not item_name then return false, "no_item_specified" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    if not inv:contains_item("main", item_name .. " " .. count) then
        return false, "no_" .. item_name
    end

    local removed = inv:remove_item("main", item_name .. " " .. count)
    if removed:is_empty() then
        return false, "remove_failed"
    end

    local pos = player:get_pos()
    local dir = player:get_look_dir()
    local drop_pos = vector.add(pos, vector.multiply(dir, 2))
    drop_pos.y = drop_pos.y + 1
    core.add_item(drop_pos, removed)
    return true, "dropped_" .. removed:to_string()
end

-- 从附近容器（箱子）取出物品
action_handlers["take_from_container"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local item_name = params.item
    local count = params.count or 1
    if not item_name then return false, "no_item_specified" end

    local pos = vector.round(player:get_pos())

    for x = -3, 3 do
        for y = -1, 2 do
            for z = -3, 3 do
                local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(check_pos)
                if node.name == "default:chest" or node.name == "default:chest_locked" then
                    local meta = core.get_meta(check_pos)
                    local chest_inv = meta:get_inventory()
                    local player_inv = player:get_inventory()

                    if chest_inv:contains_item("main", item_name .. " " .. count) then
                        local removed = chest_inv:remove_item("main", item_name .. " " .. count)
                        local leftover = player_inv:add_item("main", removed)
                        if not leftover:is_empty() then
                            chest_inv:add_item("main", leftover)
                            return false, "player_inventory_full"
                        end
                        return true, "took_" .. removed:to_string()
                    end
                end
            end
        end
    end

    return false, "item_not_in_nearby_container"
end

-- 使用/交互节点（右键点击：门、按钮、熔炉、箱子等）
action_handlers["use_node"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local target = params.target
    local pos

    if target and target.x then
        pos = vector.new(target.x, target.y, target.z)
    else
        -- 使用面前的节点
        local ppos = player:get_pos()
        local dir = player:get_look_dir()
        pos = vector.round(vector.add(ppos, vector.multiply(dir, 2)))
    end

    local node = core.get_node(pos)
    if node.name == "air" or node.name == "ignore" then
        return false, "no_node_at_target"
    end

    -- 获取节点定义并调用 on_rightclick
    local def = core.registered_nodes[node.name]
    if def and def.on_rightclick then
        def.on_rightclick(pos, node, player, player:get_wielded_item(), {
            type = "node",
            under = pos,
            above = vector.new(pos.x, pos.y + 1, pos.z),
        })
        return true, "used_" .. node.name
    end

    return false, "node_not_interactable_" .. node.name
end

-- 智能放置（相对位置：above/below/front/left/right/back）
action_handlers["place_at"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local direction = params.direction or "front"
    local node_name = params.node or params.item or params.block
    if not node_name then return false, "no_node_specified" end

    -- 如果名称不包含冒号，补上 default: 前缀
    if not node_name:find(":") then
        node_name = "default:" .. node_name
    end

    local place_pos
    local look_dir = player:get_look_dir()
    local front = vector.new(math.floor(look_dir.x + 0.5), 0, math.floor(look_dir.z + 0.5))

    if direction == "above" or direction == "up" then
        place_pos = vector.new(pos.x, pos.y + 2, pos.z)
    elseif direction == "below" or direction == "down" then
        place_pos = vector.new(pos.x, pos.y - 1, pos.z)
    elseif direction == "front" or direction == "forward" then
        place_pos = vector.add(pos, front)
    elseif direction == "back" or direction == "behind" then
        place_pos = vector.subtract(pos, front)
    elseif direction == "left" then
        place_pos = vector.new(pos.x + front.z, pos.y, pos.z - front.x)
    elseif direction == "right" then
        place_pos = vector.new(pos.x - front.z, pos.y, pos.z + front.x)
    else
        -- 直接使用坐标
        if params.target and params.target.x then
            place_pos = vector.new(params.target.x, params.target.y, params.target.z)
        else
            place_pos = vector.add(pos, front)
        end
    end

    local existing = core.get_node(place_pos)
    if existing.name ~= "air" then
        return false, "position_occupied_by_" .. existing.name
    end

    -- 检查库存中是否有此方块
    local inv = player:get_inventory()
    if inv and inv:contains_item("main", node_name) then
        inv:remove_item("main", node_name)
    end

    core.set_node(place_pos, {name = node_name})
    return true, "placed_" .. node_name .. "_" .. direction
end

-- 向下挖掘（脚下方块）
action_handlers["dig_down"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local depth = params.depth or 1

    local dug = 0
    for y = 1, depth do
        local dig_pos = vector.new(pos.x, pos.y - y, pos.z)
        local node = core.get_node(dig_pos)
        if node.name ~= "air" and node.name ~= "ignore" then
            if core.dig_node(dig_pos) then
                dug = dug + 1
            end
        end
    end

    return dug > 0, "dug_down_" .. dug .. "_blocks"
end

-- 向上挖掘（头顶方块）
action_handlers["dig_up"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local height = params.height or 1

    local dug = 0
    for y = 2, 1 + height do  -- 从头顶上方开始（y+2）
        local dig_pos = vector.new(pos.x, pos.y + y, pos.z)
        local node = core.get_node(dig_pos)
        if node.name ~= "air" and node.name ~= "ignore" then
            if core.dig_node(dig_pos) then
                dug = dug + 1
            end
        end
    end

    return dug > 0, "dug_up_" .. dug .. "_blocks"
end

-- 隧道挖掘（向前挖掘 2x1 通道）
action_handlers["tunnel"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local length = math.min(params.length or 3, 8)  -- 限制最大8格
    local dir = player:get_look_dir()
    -- 量化为主轴方向
    local front = vector.new(math.floor(dir.x + 0.5), 0, math.floor(dir.z + 0.5))
    if vector.length(front) < 0.5 then
        front = vector.new(1, 0, 0)
    end
    front = vector.normalize(front)

    local dug = 0
    for i = 1, length do
        local base = vector.add(pos, vector.multiply(front, i))
        -- 挖掘 2 格高通道（脚和头的高度）
        for y = 0, 1 do
            local dig_pos = vector.new(base.x, base.y + y, base.z)
            local node = core.get_node(dig_pos)
            if node.name ~= "air" and node.name ~= "ignore" then
                if core.dig_node(dig_pos) then
                    dug = dug + 1
                end
            end
        end
    end

    -- 挖完后向前移动
    if dug > 0 then
        player:add_velocity(vector.multiply(front, 3))
    end

    return dug > 0, "tunneled_" .. dug .. "_blocks"
end

-- 搭桥（向前跨越间隙，边走边放方块）
action_handlers["bridge"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local length = math.min(params.length or 4, 10)
    local block_name = params.block or params.node or "default:cobble"

    -- 如果名称不包含冒号，补上 default: 前缀
    if not block_name:find(":") then
        block_name = "default:" .. block_name
    end

    local dir = player:get_look_dir()
    local front = vector.new(math.floor(dir.x + 0.5), 0, math.floor(dir.z + 0.5))
    if vector.length(front) < 0.5 then
        front = vector.new(1, 0, 0)
    end
    front = vector.normalize(front)

    local inv = player:get_inventory()
    local placed = 0
    for i = 1, length do
        local bridge_pos = vector.add(pos, vector.multiply(front, i))
        bridge_pos.y = bridge_pos.y - 1  -- 脚下方
        local node = core.get_node(bridge_pos)
        if node.name == "air" or node.name:find("water") then
            -- 检查并消耗库存
            if inv and inv:contains_item("main", block_name) then
                inv:remove_item("main", block_name)
            end
            core.set_node(bridge_pos, {name = block_name})
            placed = placed + 1
        end
    end

    -- 向前移动
    if placed > 0 then
        player:add_velocity(vector.multiply(front, 3))
    end

    return placed > 0, "bridged_" .. placed .. "_blocks"
end

-- 向上搭建（跳跃+脚下放方块，垂直上升）
action_handlers["tower_up"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local height = math.min(params.height or 3, 10)
    local block_name = params.block or params.node or "default:cobble"

    if not block_name:find(":") then
        block_name = "default:" .. block_name
    end

    local inv = player:get_inventory()
    local placed = 0
    for y = 0, height - 1 do
        local tower_pos = vector.new(pos.x, pos.y + y, pos.z)
        local node = core.get_node(tower_pos)
        if node.name == "air" then
            if inv and inv:contains_item("main", block_name) then
                inv:remove_item("main", block_name)
            end
            core.set_node(tower_pos, {name = block_name})
            placed = placed + 1
        end
    end

    -- 将玩家移到塔顶
    if placed > 0 then
        player:set_pos(vector.new(pos.x, pos.y + placed + 0.5, pos.z))
    end

    return placed > 0, "towered_up_" .. placed .. "_blocks"
end

-- 种植（在耕地上放置种子）
action_handlers["farm_plant"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local seed = params.seed or params.item

    -- 自动搜索库存中的种子
    if not seed then
        local inv = player:get_inventory()
        if inv then
            local list = inv:get_list("main")
            if list then
                for _, stack in ipairs(list) do
                    local name = stack:get_name()
                    if name:find("seed") or name:find("sapling") then
                        seed = name
                        break
                    end
                end
            end
        end
    end

    if not seed then return false, "no_seeds" end

    -- 搜索附近的耕地/泥土
    local farmland_names = {"farming:soil_wet", "farming:soil", "default:dirt"}
    local search_radius = 5
    local planted = 0

    for x = -search_radius, search_radius do
        for z = -search_radius, search_radius do
            for _, soil_name in ipairs(farmland_names) do
                local ground_pos = vector.new(pos.x + x, pos.y - 1, pos.z + z)
                local ground = core.get_node(ground_pos)
                if ground.name == soil_name then
                    local above_pos = vector.new(pos.x + x, pos.y, pos.z + z)
                    local above = core.get_node(above_pos)
                    if above.name == "air" then
                        local inv = player:get_inventory()
                        if inv and inv:contains_item("main", seed) then
                            inv:remove_item("main", seed .. " 1")
                            core.set_node(above_pos, {name = seed})
                            planted = planted + 1
                            if planted >= (params.count or 1) then
                                return true, "planted_" .. planted .. "_" .. seed
                            end
                        else
                            return planted > 0, planted > 0 and "planted_" .. planted or "no_more_seeds"
                        end
                    end
                end
            end
        end
    end

    return planted > 0, planted > 0 and "planted_" .. planted or "no_farmland_nearby"
end

-- 收获农作物
action_handlers["farm_harvest"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = vector.round(player:get_pos())
    local search_radius = params.radius or 5

    -- 成熟作物（farming mod 中的最终生长阶段）
    local mature_crops = {
        "farming:wheat_8", "farming:cotton_8",
        "farming:carrot_8", "farming:potato_4",
        "farming:tomato_8", "farming:corn_8",
        "farming:melon_8", "farming:pumpkin_8",
        "default:blueberry_bush_leaves_with_berries",
        "default:apple",
    }

    local harvested = 0
    for x = -search_radius, search_radius do
        for y = -2, 3 do
            for z = -search_radius, search_radius do
                local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(check_pos)
                for _, crop in ipairs(mature_crops) do
                    if node.name == crop then
                        if core.dig_node(check_pos) then
                            harvested = harvested + 1
                        end
                    end
                end
            end
        end
    end

    return harvested > 0, "harvested_" .. harvested .. "_crops"
end

-- 冶炼（将物品放入熔炉冶炼）
action_handlers["smelt"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local input_item = params.item or params.input
    local fuel_item = params.fuel or "default:coal_lump"
    if not input_item then return false, "no_item_specified" end

    local pos = vector.round(player:get_pos())
    local player_inv = player:get_inventory()

    -- 搜索附近的熔炉
    for x = -3, 3 do
        for y = -1, 2 do
            for z = -3, 3 do
                local check_pos = vector.new(pos.x + x, pos.y + y, pos.z + z)
                local node = core.get_node(check_pos)
                if node.name == "default:furnace" or node.name == "default:furnace_active" then
                    local meta = core.get_meta(check_pos)
                    local furnace_inv = meta:get_inventory()

                    -- 放入原料
                    if player_inv:contains_item("main", input_item) then
                        local removed = player_inv:remove_item("main", input_item)
                        furnace_inv:add_item("src", removed)
                    else
                        return false, "no_" .. input_item .. "_in_inventory"
                    end

                    -- 放入燃料
                    if player_inv:contains_item("main", fuel_item) then
                        local removed = player_inv:remove_item("main", fuel_item)
                        furnace_inv:add_item("fuel", removed)
                    end

                    -- 启动熔炉计时器
                    local timer = core.get_node_timer(check_pos)
                    if timer and not timer:is_started() then
                        timer:start(1.0)
                    end

                    return true, "smelting_" .. input_item
                end
            end
        end
    end

    return false, "no_furnace_nearby"
end

-- 拾取附近掉落物
action_handlers["pickup_item"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local pos = player:get_pos()
    local radius = params.radius or 6
    local target_name = params.item

    -- 搜索附近的掉落物实体
    local best_obj = nil
    local best_dist = radius + 1

    for obj in core.objects_inside_radius(pos, radius) do
        if obj ~= player then
            local ent = obj:get_luaentity()
            if ent and ent.name == "__builtin:item" then
                local item_string = ent.itemstring or ""
                if not target_name or item_string:find(target_name) then
                    local dist = vector.distance(pos, obj:get_pos())
                    if dist < best_dist then
                        best_dist = dist
                        best_obj = obj
                    end
                end
            end
        end
    end

    if best_obj then
        -- 移向掉落物（触发自动拾取）
        local item_pos = best_obj:get_pos()
        local dir = vector.direction(pos, item_pos)
        dir.y = 0
        if vector.length(dir) > 0.01 then
            dir = vector.normalize(dir)
        end
        player:add_velocity(vector.multiply(dir, 5))
        local ent = best_obj:get_luaentity()
        return true, "picking_up_" .. (ent and ent.itemstring or "item")
    end

    return false, "no_items_nearby"
end

-- 查询合成配方
action_handlers["check_recipe"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local item_name = params.item
    if not item_name then return false, "no_item_specified" end

    local recipes = core.get_all_craft_recipes(item_name)
    if not recipes or #recipes == 0 then
        return false, "no_recipe_for_" .. item_name
    end

    -- 检查玩家是否有足够材料
    local inv = player:get_inventory()
    local recipe_info = {}
    for i, recipe in ipairs(recipes) do
        local can_craft = true
        local missing = {}
        if recipe.items then
            for _, mat in ipairs(recipe.items) do
                if mat and mat ~= "" then
                    if not inv:contains_item("main", mat) then
                        can_craft = false
                        table.insert(missing, mat)
                    end
                end
            end
        end
        table.insert(recipe_info, {
            method = recipe.method,
            craftable = can_craft,
            missing = table.concat(missing, ","),
        })
    end

    local any_craftable = false
    local info_str = ""
    for _, r in ipairs(recipe_info) do
        if r.craftable then any_craftable = true end
        info_str = info_str .. r.method .. ":" .. (r.craftable and "yes" or "no:" .. r.missing) .. ";"
    end

    return true, "recipes_" .. #recipes .. "_craftable_" .. (any_craftable and "yes" or "no") .. "_" .. info_str
end

-- 打击/敲击节点（punch_node）
action_handlers["punch_node"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local target = params.target
    local pos

    if target and target.x then
        pos = vector.new(target.x, target.y, target.z)
    else
        -- 面前的方块
        local ppos = player:get_pos()
        local dir = player:get_look_dir()
        pos = vector.round(vector.add(ppos, vector.multiply(dir, 2)))
    end

    local node = core.get_node(pos)
    if node.name == "air" or node.name == "ignore" then
        return false, "no_node"
    end

    core.punch_node(pos, player)
    return true, "punched_" .. node.name
end

-- 设置手持栏活动槽位
action_handlers["set_hotbar"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local slot = tonumber(params.slot)
    if not slot or slot < 1 or slot > 32 then
        return false, "invalid_slot"
    end

    player:set_wield_index(slot)
    local wielded = player:get_wielded_item()
    return true, "hotbar_slot_" .. slot .. "_" .. (wielded:is_empty() and "empty" or wielded:get_name())
end

-- 整理库存（将相同物品堆叠在一起）
action_handlers["sort_inventory"] = function(params)
    local player = get_player()
    if not player then return false, "no_player" end

    local inv = player:get_inventory()
    if not inv then return false, "no_inventory" end

    local list = inv:get_list("main")
    if not list then return false, "no_main_list" end

    -- 收集所有物品
    local items = {}
    for _, stack in ipairs(list) do
        if not stack:is_empty() then
            local name = stack:get_name()
            local count = stack:get_count()
            items[name] = (items[name] or 0) + count
        end
    end

    -- 清空并重新排列
    for i = 1, #list do
        inv:set_stack("main", i, "")
    end

    local slot = 1
    -- 按名称排序放回
    local sorted_names = {}
    for name in pairs(items) do
        table.insert(sorted_names, name)
    end
    table.sort(sorted_names)

    for _, name in ipairs(sorted_names) do
        local count = items[name]
        local def = core.registered_items[name]
        local max_stack = def and def.stack_max or 99

        while count > 0 and slot <= #list do
            local put = math.min(count, max_stack)
            inv:set_stack("main", slot, name .. " " .. put)
            count = count - put
            slot = slot + 1
        end
    end

    return true, "inventory_sorted_" .. #sorted_names .. "_types"
end

-- ============================================================
-- 执行动作
-- ============================================================
local function execute_action(action)
    local action_type = action.action or action.type
    if not action_type then
        log_error("动作缺少类型字段")
        return false, "no_action_type"
    end

    local handler = action_handlers[action_type]
    if not handler then
        log_error("未知动作类型: " .. tostring(action_type))
        return false, "unknown_action_" .. tostring(action_type)
    end

    local params = action.params or action
    local ok, success, message = pcall(handler, params)
    if not ok then
        log_error("执行动作 " .. action_type .. " 出错: " .. tostring(success))
        return false, "error_" .. action_type
    end

    return success, message
end

-- ============================================================
-- HTTP 通信
-- ============================================================

-- 发送状态给 Python 服务器并获取命令
local function send_state_to_server(game_state)
    local json_state = core.write_json(game_state)
    if not json_state then
        log_error("无法序列化游戏状态")
        return
    end

    http_api.fetch({
        url = CONFIG.server_url .. "/state",
        method = "POST",
        data = json_state,
        extra_headers = {"Content-Type: application/json"},
        timeout = 10,
    }, function(res)
        if res.succeeded and res.code == 200 then
            state.connected = true
            state.last_error = nil

            -- 解析返回的动作
            if res.data and res.data ~= "" then
                local ok, actions = pcall(core.parse_json, res.data)
                if ok and actions then
                    if actions.actions and type(actions.actions) == "table" then
                        state.pending_actions = actions.actions
                        state.action_index = 1
                        if #state.pending_actions > 0 then
                            log_info("收到 " .. #state.pending_actions .. " 个动作")
                        end
                    elseif actions.action then
                        state.pending_actions = {actions}
                        state.action_index = 1
                        log_info("收到 1 个动作")
                    end
                end
            end
        else
            if state.connected then
                log_info("Python 服务器连接断开 (code: " .. tostring(res.code) .. ")")
            end
            state.connected = false
        end
    end)
end

-- 发送动作结果
local function send_action_result(action, success, message)
    local result = {
        action = action.action or action.type,
        success = success,
        outcome = message,
        timestamp = os.time(),
    }

    local json_result = core.write_json(result)
    if not json_result then return end

    http_api.fetch({
        url = CONFIG.server_url .. "/action_result",
        method = "POST",
        data = json_result,
        extra_headers = {"Content-Type: application/json"},
        timeout = 8,
    }, function(res)
        -- 不需要处理响应
        log_debug("动作结果已发送: " .. tostring(success) .. " - " .. tostring(message))
    end)
end

-- ============================================================
-- 主循环 (globalstep)
-- ============================================================
core.register_globalstep(function(dtime)
    -- 动作执行计时器（优先执行动作）
    if #state.pending_actions > 0 and state.action_index <= #state.pending_actions then
        state.action_timer = state.action_timer + dtime
        if state.action_timer >= state.action_interval then
            state.action_timer = 0

            local action = state.pending_actions[state.action_index]
            log_info("执行动作 " .. state.action_index .. "/" .. #state.pending_actions ..
                     ": " .. tostring(action.action or action.type))

            local success, message = execute_action(action)
            log_info("结果: " .. tostring(success) .. " - " .. tostring(message))

            -- 发送结果
            send_action_result(action, success, message)

            state.action_index = state.action_index + 1

            -- 清理已完成的动作
            if state.action_index > #state.pending_actions then
                state.pending_actions = {}
                state.action_index = 0
                log_debug("所有动作已执行完毕")
            end
        end
        -- 动作执行期间不发送新状态，避免覆盖
        return
    end

    -- 状态发送计时器（仅在无待执行动作时发送）
    state.timer = state.timer + dtime
    if state.timer >= CONFIG.tick_interval then
        state.timer = 0

        local game_state = collect_game_state()
        if game_state then
            send_state_to_server(game_state)
        end
    end
end)

-- ============================================================
-- 聊天命令（用于调试）
-- ============================================================
core.register_chatcommand("agent_status", {
    description = "显示 Agent Bridge 状态",
    func = function(name, param)
        local msg = "Agent Bridge 状态:\n"
        msg = msg .. "  服务器: " .. CONFIG.server_url .. "\n"
        msg = msg .. "  已连接: " .. tostring(state.connected) .. "\n"
        msg = msg .. "  待执行动作: " .. #state.pending_actions .. "\n"
        msg = msg .. "  Tick间隔: " .. CONFIG.tick_interval .. "s\n"
        if state.last_error then
            msg = msg .. "  最后错误: " .. state.last_error .. "\n"
        end
        return true, msg
    end,
})

core.register_chatcommand("agent_test", {
    description = "测试发送状态到 Python 服务器",
    func = function(name, param)
        local game_state = collect_game_state()
        if game_state then
            send_state_to_server(game_state)
            return true, "已发送状态到 " .. CONFIG.server_url
        else
            return false, "无法收集游戏状态（玩家未找到）"
        end
    end,
})

-- ============================================================
-- 玩家加入时通知
-- ============================================================
core.register_on_joinplayer(function(player, last_login)
    local name = player:get_player_name()
    if name == CONFIG.player_name then
        log_info("目标玩家 " .. name .. " 已加入，Agent Bridge 开始工作")
        -- 立即发送一次状态
        core.after(1, function()
            local game_state = collect_game_state()
            if game_state then
                send_state_to_server(game_state)
            end
        end)
    end
end)

core.register_on_leaveplayer(function(player, timed_out)
    local name = player:get_player_name()
    if name == CONFIG.player_name then
        log_info("目标玩家 " .. name .. " 已离开")
    end
end)

-- ============================================================
-- 初始化完成
-- ============================================================
log_info("Agent Bridge 初始化完成")
log_info("  服务器地址: " .. CONFIG.server_url)
log_info("  Tick 间隔: " .. CONFIG.tick_interval .. "s")
log_info("  目标玩家: " .. CONFIG.player_name)
