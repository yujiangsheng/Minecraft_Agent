"""
情景记忆 (Episodic Memory)

作者: Jiangsheng Yu
许可证: MIT License

存储具体场景下的经验，支持标签搜索和上下文相似度检索。
使用 LRU 策略驱逐旧记忆，保持固定容量。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib


@dataclass
class Episode:
    """单个情景记忆"""
    id: str
    timestamp: str
    summary: str                      # 经验摘要
    lesson: str                       # 学到的教训
    tags: List[str]                   # 标签（用于检索）
    context: Dict[str, Any]           # 发生时的上下文
    outcome: str                      # 结果: success/failure/partial
    severity: float = 0.5             # 严重程度 (0-1)
    reusability: str = "medium"       # 可复用性: low/medium/high
    access_count: int = 0             # 访问次数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "lesson": self.lesson,
            "tags": self.tags,
            "context": self.context,
            "outcome": self.outcome,
            "severity": self.severity,
            "reusability": self.reusability,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            summary=data.get("summary", ""),
            lesson=data.get("lesson", ""),
            tags=data.get("tags", []),
            context=data.get("context", {}),
            outcome=data.get("outcome", "unknown"),
            severity=data.get("severity", 0.5),
            reusability=data.get("reusability", "medium"),
            access_count=data.get("access_count", 0)
        )


class EpisodicMemory:
    """情景记忆管理器"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.episodes: Dict[str, Episode] = {}
        self._tag_index: Dict[str, List[str]] = {}  # tag -> episode_ids
        
    def _generate_id(self, summary: str, timestamp: str) -> str:
        """生成唯一 ID"""
        content = f"{summary}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add(self, 
            summary: str,
            lesson: str,
            tags: List[str],
            context: Dict[str, Any] = None,
            outcome: str = "unknown",
            severity: float = 0.5,
            reusability: str = "medium") -> Episode:
        """添加新的情景记忆"""
        timestamp = datetime.now().isoformat()
        episode_id = self._generate_id(summary, timestamp)
        
        episode = Episode(
            id=episode_id,
            timestamp=timestamp,
            summary=summary,
            lesson=lesson,
            tags=tags,
            context=context or {},
            outcome=outcome,
            severity=severity,
            reusability=reusability
        )
        
        # 如果超过最大数量，删除最旧且访问次数最少的
        if len(self.episodes) >= self.max_size:
            self._evict_one()
        
        self.episodes[episode_id] = episode
        
        # 更新标签索引
        for tag in tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(episode_id)
        
        return episode
    
    def _evict_one(self):
        """驱逐一条记忆"""
        if not self.episodes:
            return
        
        # 按访问次数和时间排序，删除最少访问且最旧的
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: (e.access_count, e.timestamp)
        )
        
        victim = sorted_episodes[0]
        self.remove(victim.id)
    
    def remove(self, episode_id: str):
        """删除情景记忆"""
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # 从标签索引中移除
        for tag in episode.tags:
            if tag in self._tag_index and episode_id in self._tag_index[tag]:
                self._tag_index[tag].remove(episode_id)
        
        del self.episodes[episode_id]
    
    def get(self, episode_id: str) -> Optional[Episode]:
        """获取情景记忆"""
        episode = self.episodes.get(episode_id)
        if episode:
            episode.access_count += 1
        return episode
    
    def search_by_tags(self, tags: List[str], top_k: int = 5) -> List[Episode]:
        """按标签搜索"""
        # 计算每个 episode 匹配的标签数
        scores: Dict[str, int] = {}
        
        for tag in tags:
            for episode_id in self._tag_index.get(tag, []):
                scores[episode_id] = scores.get(episode_id, 0) + 1
        
        # 按匹配数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for episode_id in sorted_ids[:top_k]:
            episode = self.get(episode_id)
            if episode:
                results.append(episode)
        
        return results
    
    def search_by_context(self, 
                          current_state: Dict[str, Any], 
                          top_k: int = 5,
                          similarity_fn=None) -> List[Episode]:
        """按上下文相似度搜索"""
        if similarity_fn is None:
            # 默认使用简单的字段匹配
            similarity_fn = self._default_similarity
        
        scores = []
        for episode in self.episodes.values():
            score = similarity_fn(current_state, episode.context)
            scores.append((episode, score))
        
        # 按相似度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for episode, score in scores[:top_k]:
            episode.access_count += 1
            results.append(episode)
        
        return results
    
    def _default_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """默认相似度计算"""
        if not state1 or not state2:
            return 0.0
        
        # 简单的 Jaccard 相似度
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        if not union:
            return 0.0
        
        matching_values = 0
        for key in intersection:
            if state1.get(key) == state2.get(key):
                matching_values += 1
        
        return matching_values / len(union)
    
    def get_all(self) -> List[Episode]:
        """获取所有情景记忆"""
        return list(self.episodes.values())
    
    def get_recent(self, n: int = 10) -> List[Episode]:
        """获取最近的 n 条记忆"""
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return sorted_episodes[:n]
    
    def get_by_outcome(self, outcome: str) -> List[Episode]:
        """按结果获取记忆"""
        return [e for e in self.episodes.values() if e.outcome == outcome]
    
    def save(self, filepath: str):
        """保存到文件"""
        data = {
            "episodes": [e.to_dict() for e in self.episodes.values()],
            "max_size": self.max_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.max_size = data.get("max_size", 100)
        self.episodes.clear()
        self._tag_index.clear()
        
        for episode_data in data.get("episodes", []):
            episode = Episode.from_dict(episode_data)
            self.episodes[episode.id] = episode
            
            for tag in episode.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = []
                self._tag_index[tag].append(episode.id)
    
    def to_prompt_format(self, episodes: List[Episode] = None) -> List[Dict[str, Any]]:
        """转换为 prompt 格式"""
        if episodes is None:
            episodes = self.get_recent(5)
        
        return [
            {
                "id": e.id,
                "summary": e.summary,
                "lesson": e.lesson,
                "tags": e.tags,
                "outcome": e.outcome
            }
            for e in episodes
        ]
