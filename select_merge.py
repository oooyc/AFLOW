# run_real_analysis.py
import os
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import deque
# 尝试导入，如果失败则设置标志位
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDING_AVAILABLE = True
except ImportError:
    print("⚠️ `sentence-transformers` or `scikit-learn` not found. Running in fallback mode.")
    EMBEDDING_AVAILABLE = False
    # 定义缺失的类以避免NameError
    class TfidfVectorizer: pass
    class SentenceTransformer: pass
    def cosine_similarity(a, b): return [[0.0]]

# ===================================================================
# 关键配置区域 - 请务必修改！
# ===================================================================

# ❗❗❗ 请将此路径修改为您机器上 `workspace_before/MATH` 的绝对路径或相对路径
# 例如: "/home/user/project/a-flow/workspace_before/MATH"
# 或者: "C:/Users/YourUser/Documents/a-flow/workspace_before/MATH"
FULL_PATH_TO_WORKSPACE = "./workspace_before/MATH" 

# --- 分析参数 ---
# 初始候选池的大小 (选择Top N个工作流进行分析)
CANDIDATE_POOL_SIZE = 10

# 相似度阈值 (高于此值的工作流被视为一个家族)
SIMILARITY_THRESHOLD = 0.65

# ===================================================================
# 您的所有工具类代码粘贴在这里
# ===================================================================

# [在此处粘贴您之前提供的所有类: PromptSemanticAnalyzer, FailureAnalyzer, 
#  ExperienceAnalyzer, WorkflowStructureAnalyzer, WorkflowSimilarity, 
#  GraphUtils, DiversityAwareSelector, 以及两个加载函数]
# 为了使回复简洁，此处省略粘贴，但请确保您的脚本中包含它们。
# ... 假设所有类已粘贴 ...
class PromptSemanticAnalyzer:
    """基于语义的提示词分析器"""
    
    def __init__(self):
        self.model = None
        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Loaded semantic embedding model for prompt analysis")
            except Exception as e:
                print(f"⚠️ Failed to load embedding model: {e}")
                self.model = None
    
    def _extract_key_instructions(self, prompt: str) -> List[str]:
        """提取提示词中的关键指令"""
        # 分割句子并过滤短句
        sentences = re.split(r'[.!?]', prompt)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        # 提取关键指令
        key_instructions = []
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in ["step by step", "reasoning", "think", "process"]):
                key_instructions.append("step-by-step reasoning")
            if any(kw in s_lower for kw in ["boxed", "enclose", "format", "final answer"]):
                key_instructions.append("boxed answer format")
            if any(kw in s_lower for kw in ["review", "check", "verify", "correct"]):
                key_instructions.append("solution review")
            if any(kw in s_lower for kw in ["edge case", "missing case", "boundary", "special case"]):
                key_instructions.append("edge case handling")
            if "problem" in s_lower and "solution" in s_lower:
                key_instructions.append("problem-solution context")
        
        return list(set(key_instructions))
    
    def get_prompt_semantic_similarity(self, prompt_a: str, prompt_b: str) -> float:
        """使用语义模型计算提示词相似度（如果可用）"""
        if not self.model:
            # 回退到基于规则的特征匹配
            return self._get_prompt_feature_similarity(prompt_a, prompt_b)
        
        try:
            # 编码关键指令
            instructions_a = self._extract_key_instructions(prompt_a)
            instructions_b = self._extract_key_instructions(prompt_b)
            
            if not instructions_a or not instructions_b:
                # 如果没有提取到关键指令，直接比较整个prompt
                emb_a = self.model.encode([prompt_a])
                emb_b = self.model.encode([prompt_b])
                return cosine_similarity(emb_a, emb_b)[0][0]
            
            # 比较关键指令的语义相似度
            emb_a = self.model.encode(instructions_a)
            emb_b = self.model.encode(instructions_b)
            
            # 计算最大匹配相似度
            sim_matrix = cosine_similarity(emb_a, emb_b)
            avg_sim = np.mean(np.max(sim_matrix, axis=1))
            return avg_sim
        except Exception as e:
            print(f"⚠️ Semantic similarity calculation failed: {e}")
            return self._get_prompt_feature_similarity(prompt_a, prompt_b)
    
    def _get_prompt_feature_similarity(self, prompt_a: str, prompt_b: str) -> float:
        """基于规则的提示词特征相似度（回退方案）"""
        features_a = set(self._extract_key_instructions(prompt_a))
        features_b = set(self._extract_key_instructions(prompt_b))
        
        if not features_a and not features_b:
            return 1.0
        if not features_a or not features_b:
            return 0.0
        
        return len(features_a & features_b) / len(features_a | features_b)
    
class FailureAnalyzer:
    """失败模式多维分析器 - 完整且鲁棒的版本"""
    
    ERROR_TYPES = {
        "reasoning_error": [
            "incorrect logic", "flawed reasoning", "wrong approach", 
            "misunderstood problem", "invalid assumption", "logical error"
        ],
        "calculation_error": [
            "arithmetic error", "computation mistake", "algebra error",
            "numerical error", "calculation mistake", "mathematical inaccuracy"
        ],
        "omission_error": [
            "missing case", "forgot", "did not consider", "overlooked",
            "edge case", "boundary condition", "special case"
        ],
        "format_error": [
            "incorrect format", "not boxed", "multiple answers",
            "poorly formatted", "ambiguous output", "formatting issue"
        ],
        "extraction_failure": [
            "extract failed", "no boxed answer", "could not parse",
            "answer not found", "regex failed", "parsing issue"
        ]
    }
    
    @staticmethod
    def _normalize(text: Optional[str]) -> str:
        """安全地规范化文本，处理None值和非字符串类型"""
        if text is None:
            return ""
        return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())
    
    def _detect_error_type(self, justification: Optional[str]) -> str:
        """从评分理由中识别错误类型，安全处理空值"""
        text = self._normalize(justification)
        if not text.strip():  # 如果是空字符串
            return "unknown_error"
        
        for err_type, keywords in self.ERROR_TYPES.items():
            if any(k in text for k in keywords):
                return err_type
        return "unknown_error"
    
    def _analyze_extraction_robustness(self, extract_code: Optional[str]) -> Dict[str, bool]:
        """分析答案提取代码的鲁棒性，安全处理空值"""
        code = extract_code or ""
        return {
            "uses_boxed_regex": "boxed" in code and "re." in code,
            "handles_multiple_boxed": "boxed_matches[-1]" in code or "all" in code,
            "has_fallback": "sentences" in code or "split" in code,
            "regex_too_strict": r"\\boxed{([^}]*)}" in code,
            "regex_robust": r"\\\\boxed{((?:[^{}]|\{[^{}]*\})*)}" in code
        }
    
    def _assess_rating_consistency(self, rating_score: Optional[int], 
                                  extracted_output: Optional[str], 
                                  right_answer: Optional[str]) -> str:
        """评估评分是否与最终结果一致，安全处理空值"""
        # 安全转换评分
        try:
            score = int(rating_score) if rating_score is not None else 5
        except (TypeError, ValueError):
            score = 5
            
        # 简化版答案等价判断（安全处理）
        a_clean = re.sub(r'[^0-9./\\-]', '', str(extracted_output).lower()) if extracted_output else ""
        b_clean = re.sub(r'[^0-9./\\-]', '', str(right_answer).lower()) if right_answer else ""
        is_correct = a_clean.strip() == b_clean.strip()
        
        if score >= 8 and not is_correct:
            return "high_score_but_wrong"
        if score < 5 and is_correct:
            return "low_score_but_correct"
        return "consistent"
    
    def _get_empty_failure_signature(self) -> Dict:
        """返回一个空的、结构完整的失败签名"""
        return {
            "failure_modes": Counter(),
            "extraction_issues": 0,
            "rating_problems": 0,
            "common_error_type": "unknown",
            "requires_review": False,
            "requires_extraction_fix": False,
            "requires_rating_fix": False
        }
    
    def extract_failure_signature(self, failure_log: Optional[List[Dict]]) -> Dict:
        """
        从失败日志列表中提取失败模式签名，安全处理各种异常情况
        """
        # 处理空输入
        if not failure_log or not isinstance(failure_log, list):
            return self._get_empty_failure_signature()
        
        failure_modes = Counter()
        extraction_issues = 0
        rating_problems = 0
        extract_code_sample = None
        
        for item in failure_log:
            # 安全获取评分理由（多层嵌套检查）
            justification = None
            try:
                # 多层安全访问
                if "intermediate_eval" in item and item["intermediate_eval"]:
                    if isinstance(item["intermediate_eval"], list) and len(item["intermediate_eval"]) > 0:
                        if "input_rating" in item["intermediate_eval"][0]:
                            if "justification" in item["intermediate_eval"][0]["input_rating"]:
                                justification = item["intermediate_eval"][0]["input_rating"]["justification"]
            except (TypeError, IndexError, KeyError, AttributeError):
                pass
            
            # 分析错误类型（安全处理）
            error_type = self._detect_error_type(justification)
            failure_modes[error_type] += 1
            
            # 安全获取提取代码
            if extract_code_sample is None and "extract_answer_code" in item:
                extract_code_sample = item["extract_answer_code"]
            
            # 安全获取评分分数
            rating_score = None
            try:
                if "intermediate_eval" in item and item["intermediate_eval"]:
                    if isinstance(item["intermediate_eval"], list) and len(item["intermediate_eval"]) > 0:
                        if "input_rating" in item["intermediate_eval"][0]:
                            if "score" in item["intermediate_eval"][0]["input_rating"]:
                                rating_score = item["intermediate_eval"][0]["input_rating"]["score"]
            except (TypeError, IndexError, KeyError, AttributeError):
                pass
            
            # 评估评分一致性
            if "right_answer" in item and "extracted_output" in item:
                rating_consistency = self._assess_rating_consistency(
                    rating_score, item["extracted_output"], item["right_answer"]
                )
                if rating_consistency == "high_score_but_wrong":
                    rating_problems += 1
        
        # 分析提取代码
        extraction_analysis = self._analyze_extraction_robustness(extract_code_sample)
        if not extraction_analysis["uses_boxed_regex"]:
            extraction_issues += 10
        elif not extraction_analysis["has_fallback"]:
            extraction_issues += 3
        elif not extraction_analysis["regex_robust"]:
            extraction_issues += 1
        
        return {
            "failure_modes": failure_modes,
            "extraction_issues": extraction_issues,
            "rating_problems": rating_problems,
            "common_error_type": failure_modes.most_common(1)[0][0] if failure_modes else "unknown",
            "requires_review": any(t in ["reasoning_error", "omission_error"] for t in failure_modes),
            "requires_extraction_fix": extraction_issues > 0,
            "requires_rating_fix": rating_problems > len(failure_log) * 0.3 if failure_log else False
        }
    
    # 👇 这是之前遗漏的关键方法 👇
    def calculate_failure_similarity(self, sig_a: Dict, sig_b: Dict) -> float:
        """计算两个失败签名的相似度"""
        # 1. 主要错误类型匹配度
        type_match = 1.0 if sig_a["common_error_type"] == sig_b["common_error_type"] else 0.0
        
        # 2. 错误模式分布相似度 (Jaccard)
        modes_a = set(sig_a["failure_modes"].keys())
        modes_b = set(sig_b["failure_modes"].keys())
        mode_jaccard = len(modes_a & modes_b) / len(modes_a | modes_b) if (modes_a | modes_b) else 0.0
        
        # 3. 是否都需要 review
        review_needed = 1.0 if sig_a["requires_review"] == sig_b["requires_review"] else 0.5
        
        # 4. 提取问题相似度
        extraction_sim = 1.0 if (
            sig_a["requires_extraction_fix"] == sig_b["requires_extraction_fix"]
        ) else 0.0
        
        return 0.4 * type_match + 0.3 * mode_jaccard + 0.2 * review_needed + 0.1 * extraction_sim
    

class ExperienceAnalyzer:
    """基于语义的体验分析器"""
    
    DESIGN_PATTERNS = {
        "multi_stage_refinement": [
            "review", "verify", "check", "correct", "refine", "critical", 
            "after", "then", "sequential", "two-step", "three-step", "final verification"
        ],
        "self_consistency_robustness": [
            "scensemble", "self-consistency", "voting", "majority", "ensemble", 
            "multiple solutions", "reduce errors", "random errors", "independent paths"
        ],
        "computational_verification": [
            "programmer", "code execution", "generate and execute", "validate", 
            "computational validation", "execute code", "run code"
        ],
        "prompt_engineering": [
            "prompt", "instruction", "require", "emphasize", "boxed answer", 
            "step-by-step", "format", "comma-separation", "strict boxing"
        ],
        "error_handling": [
            "error handling", "non-empty", "ensure output", "fallback", "default"
        ],
        "conditional_execution": [
            "conditional", "if", "only if", "depending on", "dynamic", "contextual"
        ],
        "multi_solution_generation": [
            "generate three", "multiple candidate", "independent reasoning", 
            "combine solutions", "select most consistent"
        ]
    }
    
    def __init__(self):
        self.failure_analyzer = FailureAnalyzer()
        self.prompt_analyzer = PromptSemanticAnalyzer()
    
    def _normalize_text(self, text: str) -> str:
        return re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    
    def extract_design_patterns(self, modification: str) -> Set[str]:
        """从修改描述中提取设计模式（增强版）"""
        if not modification or not isinstance(modification, str):
            return set()
        
        text = self._normalize_text(modification)
        matched = set()
        
        # 1. 基础模式匹配
        for pattern, keywords in self.DESIGN_PATTERNS.items():
            if any(k in text for k in keywords):
                matched.add(pattern)
        
        # 2. 检测多阶段精炼的深度（关键改进）
        if "review" in text or "verify" in text or "check" in text:
            # 检测阶段数量
            stage_keywords = [
                ("first", 1), ("initial", 1),
                ("second", 2), ("review", 2),
                ("third", 3), ("verify", 3), ("final", 3)
            ]
            
            max_stage = 1
            for keyword, stage in stage_keywords:
                if keyword in text:
                    max_stage = max(max_stage, stage)
            
            # 添加阶段深度信息
            if max_stage > 1:
                matched.add(f"multi_stage_refinement_depth{max_stage}")
        
        # 3. 检测自洽性集成的规模
        if "ensemble" in text or "voting" in text:
            if "three" in text or "3" in text:
                matched.add("self_consistency_ensemble_size3")
            elif "two" in text or "2" in text:
                matched.add("self_consistency_ensemble_size2")
        
        return matched
    
    def get_workflow_signature(self, 
                              workflow_id: str, 
                              experience: Dict,
                              failure_logs: Optional[Dict[str, List[Dict]]] = None) -> Dict:
        """为工作流生成完整的行为签名"""
        if workflow_id not in experience:
            return {
                "success_patterns": Counter(),
                "failure_patterns": Counter(),
                "all_patterns": Counter(),
                "raw_modifications": [],
                "failure_signature": self.failure_analyzer.extract_failure_signature([])
            }
        
        node = experience[workflow_id]
        success_patterns = Counter()
        failure_patterns = Counter()
        all_mods = []
        
        # 分析成功历史
        for _, data in node.get("success", {}).items():
            mod = data.get("modification", "")
            all_mods.append(mod)
            patterns = self.extract_design_patterns(mod)
            success_patterns.update(patterns)
        
        # 分析失败历史
        for _, data in node.get("failure", {}).items():
            mod = data.get("modification", "")
            all_mods.append(mod)
            patterns = self.extract_design_patterns(mod)
            failure_patterns.update(patterns)
        
        total_patterns = success_patterns + failure_patterns
        
        # 分析失败日志（如果提供）
        failure_log = failure_logs.get(workflow_id, []) if failure_logs else []
        failure_sig = self.failure_analyzer.extract_failure_signature(failure_log)
        
        return {
            "success_patterns": success_patterns,
            "failure_patterns": failure_patterns,
            "all_patterns": total_patterns,
            "raw_modifications": all_mods,
            "failure_signature": failure_sig
        }
    
    def calculate_design_similarity(self, sig_a: Dict, sig_b: Dict) -> float:
        """计算两个工作流设计意图的相似度"""
        def counter_jaccard(c1: Counter, c2: Counter) -> float:
            if not c1 and not c2:
                return 0.5
            if not c1 or not c2:
                return 0.0
            set1, set2 = set(c1.keys()), set(c2.keys())
            return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
        
        def counter_weighted_sim(c1: Counter, c2: Counter) -> float:
            if not c1 and not c2:
                return 0.5
            if not c1 or not c2:
                return 0.0
            weight_inter = sum(c1[k] * c2[k] for k in set(c1.keys()) & set(c2.keys()))
            weight_union = sum((c1[k])**2 for k in c1) + sum((c2[k])**2 for k in c2)
            return weight_inter / (weight_union**0.5 + 1e-8)
        
        # 1. 成功模式相似度（最重要：它们如何成功）
        success_sim = counter_weighted_sim(sig_a["success_patterns"], sig_b["success_patterns"])
        
        # 2. 失败模式相似度（共性弱点）
        failure_sim = counter_jaccard(sig_a["failure_patterns"], sig_b["failure_patterns"])
        
        # 3. 总体模式相似度（设计哲学）
        total_sim = counter_jaccard(sig_a["all_patterns"], sig_b["all_patterns"])
        
        # 4. 失败根因相似度
        failure_sig_sim = self.failure_analyzer.calculate_failure_similarity(
            sig_a["failure_signature"], sig_b["failure_signature"]
        )
        
        # 加权融合：成功意图最重要，失败根因次之
        return (
            0.4 * success_sim +
            0.3 * total_sim +
            0.2 * failure_sig_sim +
            0.1 * failure_sim
        )
    
class WorkflowStructureAnalyzer:
    """工作流结构分析器 - 改进版"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words=None, 
            tokenizer=self._custom_tokenizer,
            token_pattern=None
        )

    def extract_workflow_topology(self, code: str) -> Dict:
        """分析工作流的拓扑结构特征"""
        # 1. 节点类型统计
        node_types = re.findall(r'self\.\w+\s*=\s*operator\.(\w+)', code)
        node_type_counter = Counter(node_types)
        
        # 2. 节点调用序列
        call_sequence = re.findall(r'await\s+self\.(\w+)\(', code)
        
        # 3. 依赖关系分析（识别阶段）
        stages = []
        current_stage = []
        
        for line in code.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 检测节点调用
            if 'await' in line and 'self.' in line:
                if current_stage:
                    stages.append(current_stage)
                    current_stage = []
            
            # 检测变量赋值（构建依赖）
            if '=' in line and 'await' not in line:
                var = line.split('=')[0].strip()
                current_stage.append(var)
        
        if current_stage:
            stages.append(current_stage)
        
        return {
            "node_types": dict(node_type_counter),
            "call_sequence": call_sequence,
            "num_stages": len(stages),
            "max_parallelism": max(len(stage) for stage in stages) if stages else 1,
            "stage_dependencies": [len(stage) for stage in stages]
        }
    
    def _extract_workflow_topology(self, code: str) -> Dict:
        """从代码字符串中提取工作流拓扑结构特征"""
        # 1. 节点类型统计（operator类型）
        node_types = re.findall(
            r'self\.\w+\s*=\s*operator\.(\w+)', 
            code
        )
        node_type_counter = Counter(node_types)
        
        # 2. 节点调用序列（执行顺序）
        call_sequence = re.findall(
            r'await\s+self\.(\w+)\(|\.run\(', 
            code
        )
        
        # 3. 阶段分析（基于代码结构）
        stages = []
        current_stage = []
        
        # 简化的阶段检测（可根据实际代码结构调整）
        for line in code.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 检测节点调用（表示新阶段开始）
            if re.search(r'await\s+self\.\w+\(', line):
                if current_stage:
                    stages.append(current_stage)
                    current_stage = []
                current_stage.append("call")
            
            # 检测变量赋值（构建依赖）
            elif '=' in line and 'operator' in line:
                var = re.search(r'self\.(\w+)\s*=', line)
                if var:
                    current_stage.append(f"node_{var.group(1)}")
        
        if current_stage:
            stages.append(current_stage)
        
        return {
            "node_types": dict(node_type_counter),
            "call_sequence": call_sequence,
            "num_stages": len(stages),
            "max_parallelism": max(len(stage) for stage in stages) if stages else 1,
            "stage_dependencies": [len(stage) for stage in stages]
        }
    
    def calculate_topology_similarity(self, code_a: str, code_b: str) -> float:
        """计算两个工作流代码的拓扑结构相似度"""
        topo_a = self._extract_workflow_topology(code_a)
        topo_b = self._extract_workflow_topology(code_b)
        
        # 1. 节点类型分布相似度
        types_a = Counter(topo_a["node_types"])
        types_b = Counter(topo_b["node_types"])
        total_a = sum(types_a.values())
        total_b = sum(types_b.values())
        
        if total_a == 0 or total_b == 0:
            return 0.0
            
        type_sim = sum(min(types_a[k], types_b[k]) for k in set(types_a) & set(types_b)) / min(total_a, total_b)
        
        # 2. 阶段数相似度（关键指标）
        stage_diff = abs(topo_a["num_stages"] - topo_b["num_stages"])
        stage_sim = max(0, 1.0 - stage_diff * 0.2)  # 每差一个阶段扣 0.2
        
        # 3. 并行度相似度
        parallel_diff = abs(topo_a["max_parallelism"] - topo_b["max_parallelism"])
        parallel_sim = max(0, 1.0 - parallel_diff * 0.3)
        
        # 4. 调用序列编辑距离
        seq_sim = self._edit_distance_similarity(
            " -> ".join(topo_a["call_sequence"]), 
            " -> ".join(topo_b["call_sequence"])
        )
        
        return 0.3 * type_sim + 0.4 * stage_sim + 0.2 * parallel_sim + 0.1 * seq_sim

    def _edit_distance_similarity(self, str1: str, str2: str) -> float:
        """使用编辑距离计算字符串相似度"""
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        max_len = max(len(str1), len(str2))
        return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 1.0


    
    def _custom_tokenizer(self, text: str) -> List[str]:
        tokens = []
        
        # 1. 提取算子使用（带计数信息）
        operators = re.findall(
            r'self\.(custom|programmer|sc_ensemble|test|review|format|validator)', 
            text
        )
        # 添加带计数的token（关键改进）
        for op in set(operators):
            count = operators.count(op)
            tokens.append(f"{op}_x{count}")
        
        # 2. 提取提示词使用（带顺序信息）
        prompt_keywords = re.findall(r'prompt_custom\.([A-Z_]+)', text)
        for i, kw in enumerate(prompt_keywords):
            tokens.append(f"prompt_{kw.lower()}_pos{i}")
        
        # 3. 添加控制流结构序列（关键改进）
        control_flow = []
        if re.search(r'for\s+\w+\s+in\s+range', text): control_flow.append('for_loop')
        if re.search(r'while\s+', text): control_flow.append('while_loop')
        if 'if' in text: control_flow.append('if')
        if 'else' in text: control_flow.append('else')
        tokens.append("control_flow_" + "_".join(control_flow))
        
        # 4. 添加阶段数量信息（关键改进）
        if re.search(r'await\s+self\.custom.*?await\s+self\.custom', text):
            tokens.append("multi_stage")
            # 检测阶段数量
            stage_count = len(re.findall(r'await\s+self\.\w+', text))
            tokens.append(f"stage_count_{min(stage_count, 5)}")  # 限制最大计数
        
        return tokens
    
    def _remove_strings_and_comments(self, code: str) -> str:
        """更鲁棒的字符串和注释移除"""
        # 移除三引号字符串（支持多行）
        code = re.sub(r'"""[^"]*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''[^']*?'''", '', code, flags=re.DOTALL)
        
        # 移除双引号字符串（支持转义）
        code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '', code)
        code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", '', code)
        
        # 移除 # 开头的注释
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # 移除成员变量赋值（保留结构）
        code = re.sub(r'self\.[a-z_]+ = [^\n]+', '', code)
        
        return code
    
    def extract_workflow_features(self, workflow_code: str) -> str:
        """提取工作流的结构特征"""
        simplified = self._remove_strings_and_comments(workflow_code)
        return simplified
    
    def calculate_structural_similarity(self, code_a: str, code_b: str) -> float:
        """计算结构相似度"""
        feat_a = self.extract_workflow_features(code_a)
        feat_b = self.extract_workflow_features(code_b)
        
        # 使用TF-IDF + 余弦相似度
        tfidf_matrix = self.vectorizer.fit_transform([feat_a, feat_b])
        structural_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # 分析控制流复杂度（作为辅助信号）
        complexity_a = self._analyze_control_flow_complexity(code_a)
        complexity_b = self._analyze_control_flow_complexity(code_b)
        complexity_sim = 1.0 - abs(complexity_a - complexity_b) / max(complexity_a + complexity_b, 1)
        
        return 0.7 * structural_sim + 0.3 * complexity_sim
    
    def _analyze_control_flow_complexity(self, code: str) -> float:
        """分析控制流复杂度（简化版）"""
        complexity = 0.0
        if re.search(r'for\s+\w+\s+in\s+', code):
            complexity += 0.3
        if re.search(r'while\s+', code):
            complexity += 0.4
        if re.search(r'if\s+', code):
            complexity += 0.2
            if re.search(r'elif\s+', code):
                complexity += 0.1
        if re.search(r'try\s*:', code):
            complexity += 0.2
        return min(1.0, complexity)
    
class WorkflowSimilarity:
    """综合工作流相似度计算器"""
    
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.structure_analyzer = WorkflowStructureAnalyzer()
        self.experience_analyzer = ExperienceAnalyzer()
        self.prompt_analyzer = PromptSemanticAnalyzer()
        self.failure_analyzer = FailureAnalyzer()
    
    def calculate_workflow_similarity(
        self, 
        workflow_a_id: str, 
        workflow_b_id: str, 
        processed_experience: Dict,
        failure_logs: Dict[str, List[Dict]],
        graph_utils,
        graph_path: str
    ) -> float:
        """计算两个工作流之间的综合相似度 (0-1)"""
        # 1. 获取工作流代码
        graph_a = graph_utils.extract_solve_graph(
            graph_utils.read_graph_files(workflow_a_id, graph_path)[1]
        )[0]
        graph_b = graph_utils.extract_solve_graph(
            graph_utils.read_graph_files(workflow_b_id, graph_path)[1]
        )[0]
        
        # 2. 结构相似度 (现在包含拓扑分析)
        structural_sim = self.structure_analyzer.calculate_structural_similarity(graph_a, graph_b)
        
        # 新增：拓扑结构相似度（关键指标）
        topo_sim = self.structure_analyzer.calculate_topology_similarity(graph_a, graph_b)
        
        # 3. 设计意图相似度
        sig_a = self.experience_analyzer.get_workflow_signature(
            workflow_a_id, processed_experience, failure_logs
        )
        sig_b = self.experience_analyzer.get_workflow_signature(
            workflow_b_id, processed_experience, failure_logs
        )
        design_sim = self.experience_analyzer.calculate_design_similarity(sig_a, sig_b)
        
        # 4. 提示词相似度
        _, code_a = graph_utils.read_graph_files(workflow_a_id, graph_path)
        _, code_b = graph_utils.read_graph_files(workflow_b_id, graph_path)
        prompt_a = self._extract_all_prompts(code_a)
        prompt_b = self._extract_all_prompts(code_b)
        prompt_sims = []
        for key in set(prompt_a.keys()) | set(prompt_b.keys()):
            if key in prompt_a and key in prompt_b:
                sim = self.prompt_analyzer.get_prompt_semantic_similarity(
                    prompt_a[key], prompt_b[key]
                )
                prompt_sims.append(sim)
        prompt_sim = np.mean(prompt_sims) if prompt_sims else 0.5
        
        # 5. 动态调整权重（关键改进）
        # 如果拓扑结构差异大，降低设计意图权重
        topology_diff = 1.0 - topo_sim
        if topology_diff > 0.4:  # 结构差异显著
            total_sim = (
                0.5 * structural_sim + 
                0.3 * topo_sim +     # 给拓扑更高权重
                0.1 * design_sim +   # 降低设计意图权重
                0.1 * prompt_sim
            )
        else:
            total_sim = (
                0.3 * structural_sim +
                0.3 * topo_sim +
                0.3 * design_sim +
                0.1 * prompt_sim
            )
        
        return max(0.0, min(1.0, total_sim))
    
    def _extract_all_prompts(self, code: str) -> Dict[str, str]:
        """从代码中提取所有提示词内容"""
        prompts = {}
        
        # 查找 prompt_custom 类定义
        prompt_class_match = re.search(r'class prompt_custom:\s*(.*?)(?=\n\s*\w|\Z)', code, re.DOTALL)
        if prompt_class_match:
            prompt_class = prompt_class_match.group(1)
            # 提取每个 prompt 定义
            for match in re.finditer(r'([A-Z_]+)\s*=\s*"""(.*?)"""', prompt_class, re.DOTALL):
                prompt_name = match.group(1)
                prompt_content = match.group(2).strip()
                prompts[prompt_name] = prompt_content
            
            for match in re.finditer(r'([A-Z_]+)\s*=\s*\'\'\'(.*?)\'\'\'', prompt_class, re.DOTALL):
                prompt_name = match.group(1)
                prompt_content = match.group(2).strip()
                prompts[prompt_name] = prompt_content
            
            for match in re.finditer(r'([A-Z_]+)\s*=\s*"([^"]+)"', prompt_class):
                prompt_name = match.group(1)
                prompt_content = match.group(2).strip()
                prompts[prompt_name] = prompt_content
            
            for match in re.finditer(r'([A-Z_]+)\s*=\s*\'([^\']+)\'', prompt_class):
                prompt_name = match.group(1)
                prompt_content = match.group(2).strip()
                prompts[prompt_name] = prompt_content
        
        return prompts
    
    def find_similar_workflows(
        self, 
        candidate_ids: List[str], 
        processed_experience: Dict,
        failure_logs: Dict[str, List[Dict]],
        graph_utils, 
        graph_path: str, 
        similarity_threshold: float = 0.65
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        使用“连通分量”算法在候选工作流中查找相似家族。
        此版本能正确处理“朋友的朋友”式关系，并会将所有节点都分组。
        """
        n = len(candidate_ids)
        if n == 0:
            return [], np.zeros((0, 0))
        
        # 1. 计算完整的相似度矩阵
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # (此处调用 self.calculate_workflow_similarity 来计算相似度)
                # ... 此处省略计算过程 ...
                # 假设 sim 已被计算出来
                sim = self.calculate_workflow_similarity(
                    candidate_ids[i], candidate_ids[j],
                    processed_experience, failure_logs,
                    graph_utils, graph_path
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # 2. 构建邻接表图，其中边表示相似度高于阈值
        adj = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # 3. 查找图中的所有连通分量 (这会找到所有家族，包括大小为1的)
        all_groups = []
        visited = [False] * n
        
        for i in range(n):
            if not visited[i]:
                component_indices = []
                q = deque([i]) # 使用双端队列进行广度优先搜索
                visited[i] = True
                
                while q:
                    u = q.popleft()
                    component_indices.append(u)
                    for v in adj.get(u, []): # 使用 adj.get(u, []) 保证安全
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                
                # 将索引转换回工作流ID
                all_groups.append([candidate_ids[idx] for idx in component_indices])

        # 按家族大小和成员ID排序，方便查看
        all_groups.sort(key=lambda g: (-len(g), g[0] if g else ""))
        
        return all_groups, similarity_matrix
    

class GraphUtils:
    """简化的GraphUtils，仅包含调试所需功能"""
    def __init__(self, root_path: str):
        self.root_path = root_path

    def read_graph_files(self, round_number: str, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
        with open(graph_file_path, "r", encoding="utf-8") as f:
            graph_content = f.read()
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        return re.findall(r"class Workflow:.+", graph_load, re.DOTALL)


class DiversityAwareSelector:
    """考虑多样性的选择器 - 增强版"""
    def __init__(self, root_path: str, similarity_threshold: float = 0.75):
        self.root_path = root_path
        self.similarity_threshold = similarity_threshold
        self.graph_utils = GraphUtils(root_path)
        self.workflow_similarity = WorkflowSimilarity(root_path)

    def should_merge_during_sampling(
        self, 
        selected_workflows: List[Dict], 
        processed_experience: Dict, 
        failure_logs: Dict[str, List[Dict]],
        graph_path: str
    ) -> Tuple[bool, List[List[str]], np.ndarray]:
        """判断在采样阶段是否应该合并工作流，并返回相似度矩阵"""
        workflow_ids = [wf["round"] for wf in selected_workflows]
        
        similar_groups, similarity_matrix = self.workflow_similarity.find_similar_workflows(
            workflow_ids, processed_experience, failure_logs,
            self.graph_utils, graph_path, self.similarity_threshold
        )
        
        should_merge = any(len(group) > 1 for group in similar_groups)
        return should_merge, similar_groups, similarity_matrix


def load_processed_experience(workflows_path: str) -> Dict:
    """从 workflows 目录下的 processed_experience.json 加载经验数据"""
    exp_file = os.path.join(workflows_path, "processed_experience.json")
    with open(exp_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_failure_logs(workflows_path: str) -> Dict:
    """从每个 round 目录的 log.json 加载失败日志"""
    failure_logs = {}
    for round_dir in os.listdir(workflows_path):
        if not round_dir.startswith("round_"):
            continue
        log_file = os.path.join(workflows_path, round_dir, "log.json")
        if not os.path.exists(log_file):
            continue
        with open(log_file, "r", encoding="utf-8") as f:
            failure_logs[round_dir.split("_")[1]] = json.load(f)
    return failure_logs



# ======================
# 真实数据加载工具
# ======================

def load_top_candidate_workflows(results_path: str, pool_size: int) -> List[Dict]:
    """
    从 results.json 加载并计算每个round的平均分，返回Top N个工作流。
    """
    print(f"\nReading scores from: {results_path}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found at {results_path}")
    
    df = pd.read_json(results_path)
    
    # 计算每个 round 的平均分
    avg_scores = df.groupby('round')['score'].mean().reset_index()
    
    # 排序并选出 Top N
    top_workflows_df = avg_scores.sort_values(by='score', ascending=False).head(pool_size)
    
    # 转换为所需的字典列表格式
    candidates = top_workflows_df.to_dict('records')
    # 确保 round 是字符串
    for c in candidates:
        c['round'] = str(c['round'])
        
    return candidates

def load_all_failure_logs(workflows_path: str) -> Dict[str, List[Dict]]:
    """从每个 round 目录的 log.json 加载所有失败日志"""
    print(f"Loading failure logs from subdirectories in: {workflows_path}")
    failure_logs = {}
    if not os.path.isdir(workflows_path):
        return {}
        
    for item in os.listdir(workflows_path):
        if item.startswith("round_"):
            round_id = item.split("_")[1]
            log_file = os.path.join(workflows_path, item, "log.json")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        failure_logs[round_id] = json.load(f)
                except json.JSONDecodeError:
                    print(f"  - Warning: Could not decode JSON from {log_file}")
                    failure_logs[round_id] = []
    return failure_logs

# ======================
# 核心演示逻辑 (与之前相同)
# ======================

def select_representatives(
    candidate_workflows: List[Dict],
    similar_groups: List[List[str]]
) -> List[Dict]:
    """实现“代表选举制”逻辑。"""
    candidates_map = {str(wf["round"]): wf for wf in candidate_workflows}
    new_candidate_pool = []
    processed_ids = set()

    print("\n--- Step 2.1: Electing Representatives for each Family ---")
    if not similar_groups:
        print("No families found to elect from.")

    for group in similar_groups:
        group_candidates = [candidates_map[gid] for gid in group if gid in candidates_map]
        if not group_candidates: continue
        representative = max(group_candidates, key=lambda x: x["score"])
        new_candidate_pool.append(representative)
        for gid in group:
            processed_ids.add(gid)
        print(f"\nFamily {group}:")
        # 修改后（正确）
        # 先生成成员描述列表
        member_descriptions = [
            f"Round {c['round']} (Score: {c['score']:.4f})" 
            for c in group_candidates
        ]

        # 再打印
        print(f"  - Members: {member_descriptions}")
        # 在外部增加了一对花括号 {}
        print(f"  - 👑 Elected Representative: Round {representative['round']} (Score: {representative['score']:.4f})")

    print("\n--- Step 2.2: Adding Independent Workflows ---")
    independent_count = 0
    for wf in candidate_workflows:
        if str(wf["round"]) not in processed_ids:
            new_candidate_pool.append(wf)
            independent_count += 1
    print(f"Added {independent_count} independent workflow(s).")
    
    new_candidate_pool.sort(key=lambda x: x["score"], reverse=True)
    return new_candidate_pool

# ======================
# 核心逻辑升级区域
# ======================

def compute_selection_probabilities(items: List[Dict], alpha=0.2, lambda_=0.3) -> np.ndarray:
    """
    为一组带分数的项计算选择概率。
    直接复用并适配 DataUtils 中的逻辑。
    """
    scores = np.array([item["score"] * 100 for item in items], dtype=np.float64)
    n = len(scores)
    if n == 0: return np.array([])

    uniform_prob = np.full(n, 1.0 / n)
    
    # Softmax on scores
    exp_weights = np.exp(alpha * (scores - np.max(scores)))
    score_prob = exp_weights / np.sum(exp_weights)
    
    # Mix uniform and score-based probabilities
    mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob
    return mixed_prob / np.sum(mixed_prob)


def pool_family_experience(
    family_group: List[str],
    all_experience: Dict,
    all_failure_logs: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    为指定的家族池化经验和失败日志。
    """
    pooled_experience = {"success": {}, "failure": {}}
    pooled_logs = []
    
    modification_set = set() # 防止重复添加完全相同的修改描述
    
    for member_id in family_group:
        # 1. 池化 processed_experience
        member_exp = all_experience.get(member_id)
        if member_exp:
            for succ_id, succ_data in member_exp.get("success", {}).items():
                if succ_data['modification'] not in modification_set:
                    pooled_experience["success"][f"{member_id}-{succ_id}"] = succ_data
                    modification_set.add(succ_data['modification'])
            for fail_id, fail_data in member_exp.get("failure", {}).items():
                 if fail_data['modification'] not in modification_set:
                    pooled_experience["failure"][f"{member_id}-{fail_id}"] = fail_data
                    modification_set.add(fail_data['modification'])
                    
        # 2. 池化 failure_logs
        member_logs = all_failure_logs.get(member_id)
        if member_logs:
            pooled_logs.extend(member_logs)
            
    return pooled_experience, pooled_logs


def analyze_and_present_families(
    candidate_workflows: List[Dict],
    all_groups: List[List[str]],
    all_experience: Dict,
    all_failure_logs: Dict
) -> List[Dict]:
    """
    实现“概率性代表选举”并清晰地展示所有家族（包括独立家族）。
    """
    candidates_map = {str(wf["round"]): wf for wf in candidate_workflows}
    final_representatives = []

    print("\n" + "="*23 + " Family Analysis & Selection " + "="*23)
    for i, group in enumerate(all_groups):
        print(f"\n--- Family {i+1} (Size: {len(group)}) ---")
        
        # 统一处理，不再需要 if/else
        group_candidates = [candidates_map[gid] for gid in group if gid in candidates_map]
        if not group_candidates: continue

        # 为家族池化经验
        pooled_exp, pooled_logs = pool_family_experience(
            group, all_experience, all_failure_logs
        )

        # 如果是独立家族，它就是自己的代表
        if len(group_candidates) == 1:
            representative = group_candidates[0]
            print(f"  - 🧍‍♂️ Independent Workflow: Round {representative['round']} (Score: {representative['score']:.4f})")
        else: # 如果是多成员家族，则进行概率选举
            probabilities = compute_selection_probabilities(group_candidates)
            probabilities /= probabilities.sum() # 归一化
            representative = np.random.choice(group_candidates, p=probabilities)
            
            print(f"  - Members: {group}")
            for gc in sorted(group_candidates, key=lambda x: x['score'], reverse=True):
                # 找到原始索引以匹配概率
                original_index = group_candidates.index(gc)
                prob_percent = probabilities[original_index] * 100
                marker = "★" if gc['round'] == representative['round'] else "  "
                print(f"    {marker} Round {gc['round']} (Score: {gc['score']:.4f}, Select Prob: {prob_percent:.2f}%)")
            print(f"  - 👑 Probabilistically Elected Representative: Round {representative['round']}")
        
        # 为当选的代表附加集体经验
        representative['pooled_experience'] = pooled_exp
        representative['pooled_logs'] = pooled_logs
        final_representatives.append(representative)

        print(f"  - 🧠 Attached Pooled Experience: {len(pooled_exp['success'])} success, {len(pooled_exp['failure'])} failure mods.")
        print(f"  - 📋 Attached Pooled Logs: {len(pooled_logs)} entries.")

    print("\n" + "="*70)

    final_representatives.sort(key=lambda x: x["score"], reverse=True)
    return final_representatives


# ======================
# 主函数
# ======================

def main():
    """主调试函数"""
    print("="*60)
    print("      WORKFLOW SIMILARITY & MERGE STRATEGY ANALYSIS      ")
    print("="*60)

    # --- 0. 路径和文件校验 ---
    if "YOUR_PATH" in FULL_PATH_TO_WORKSPACE:
        print("\n❌ ERROR: Please update the 'FULL_PATH_TO_WORKSPACE' variable in the script!")
        return
        
    workflows_path = os.path.join(FULL_PATH_TO_WORKSPACE, "workflows")
    results_file = os.path.join(workflows_path, "results.json")
    experience_file = os.path.join(workflows_path, "processed_experience.json")

    if not os.path.isdir(workflows_path) or not os.path.exists(results_file) or not os.path.exists(experience_file):
        print(f"\n❌ ERROR: Workspace path is invalid or missing required files.")
        print(f"Checked path: {workflows_path}")
        print(f"Ensure 'results.json' and 'processed_experience.json' exist inside it.")
        return

    try:
        # --- 1. 初始化和加载真实数据 ---
        selector = DiversityAwareSelector(root_path=FULL_PATH_TO_WORKSPACE, similarity_threshold=SIMILARITY_THRESHOLD)
        print(f"\n🚀 Initialized DiversityAwareSelector with threshold={selector.similarity_threshold}")
        
        # 加载数据
        candidate_workflows = load_top_candidate_workflows(results_file, CANDIDATE_POOL_SIZE)
        processed_experience = load_processed_experience(workflows_path)
        failure_logs = load_all_failure_logs(workflows_path)
        
        if not candidate_workflows:
            print("\n❌ ERROR: No candidate workflows could be loaded. Check 'results.json'.")
            return
            
        candidate_ids = [wf['round'] for wf in candidate_workflows]
        print(f"\nTop {len(candidate_workflows)} Candidates from `results.json`:")
        for wf in candidate_workflows:
            print(f"  - Round {wf['round']} (Avg Score: {wf['score']:.4f})")
            
        # --- 2. 运行相似度分析 ---
        print("\n--- Step 1: Running Similarity Analysis on Real Data ---")
        should_merge, similar_groups, sim_matrix = selector.should_merge_during_sampling(
            selected_workflows=candidate_workflows,
            processed_experience=processed_experience,
            failure_logs=failure_logs,
            graph_path=workflows_path
        )
        
        print("\nSimilarity Matrix:")
        header = "      " + "  ".join([f"R{id:<5}" for id in candidate_ids])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(sim_matrix):
            print(f"R{candidate_ids[i]:<5}" + "  ".join([f"{x:^6.3f}" for x in row]))
            
        print(f"\nShould merge based on threshold? 👉 {should_merge}")
        if should_merge:
            print(f"Found Similar Families (Clusters): {similar_groups}")
        else:
            print("No similar families found above the threshold.")

        # --- 3. 分析家族、概率性选举并池化经验 ---
        final_representative_pool = analyze_and_present_families(
            candidate_workflows,
            similar_groups,
            processed_experience,
            failure_logs
        )
            
        print("\n--- Step 3: Final Diversified Candidate Pool ---")
        print("This is the recommended, de-duplicated pool for your next selection step:")
        for i, wf in enumerate(final_representative_pool):
            print(f"  {i+1}. Round {wf['round']} (Score: {wf['score']:.4f})")

    except Exception as e:
        import traceback
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()