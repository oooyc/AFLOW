import json
import os
from collections import defaultdict
os.sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file
from diversity_utils import ExperienceAnalyzer
import numpy as np
from collections import Counter

class ExperienceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        # 在 ExperienceUtils 内部实例化一个分析器，方便调用
        self.experience_analyzer = ExperienceAnalyzer()

    def load_experience(self, path=None, mode: str = "Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        experience_data = defaultdict(lambda: {"score": None, "success": {}, "failure": {}})

        for round_dir in os.listdir(rounds_dir):
            if os.path.isdir(os.path.join(rounds_dir, round_dir)) and round_dir.startswith("round_"):
                round_path = os.path.join(rounds_dir, round_dir)
                try:
                    round_number = int(round_dir.split("_")[1])
                    json_file_path = os.path.join(round_path, "experience.json")
                    if os.path.exists(json_file_path):
                        data = read_json_file(json_file_path, encoding="utf-8")
                        father_node = data["father node"]

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = data["before"]

                        if data["succeed"]:
                            experience_data[father_node]["success"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        else:
                            experience_data[father_node]["failure"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                except Exception as e:
                    logger.info(f"Error processing {round_dir}: {str(e)}")

        experience_data = dict(experience_data)

        output_path = os.path.join(rounds_dir, "processed_experience.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(experience_data, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Processed experience data saved to {output_path}")
        return experience_data

    def format_experience(self, processed_experience, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            experience = f"Original Score: {experience_data['score']}\n"
            experience += "These are some conclusions drawn from experience:\n\n"
            for key, value in experience_data["failure"].items():
                experience += f"-Absolutely prohibit {value['modification']} (Score: {value['score']})\n"
            for key, value in experience_data["success"].items():
                experience += f"-Absolutely prohibit {value['modification']} \n"
            experience += "\n\nNote: Take into account past failures and avoid repeating the same mistakes, as these failures indicate that these approaches are ineffective. You must fundamentally change your way of thinking, rather than simply using more advanced Python syntax like for, if, else, etc., or modifying the prompt."
        else:
            experience = f"No experience data found for round {sample_round}."
        return experience

    def check_modification(self, processed_experience, modification, sample_round):
        experience_data = processed_experience.get(int(sample_round))
        if experience_data:
            for key, value in experience_data["failure"].items():
                if value["modification"] == modification:
                    return False
            for key, value in experience_data["success"].items():
                if value["modification"] == modification:
                    return False
            return True
        else:
            return True  # 如果 experience_data 为空，也返回 True

    def create_experience_data(self, sample, modification):
        return {
            "father node": sample["round"],
            "modification": modification,
            "before": sample["score"],
            "after": None,
            "succeed": None,
        }

    def update_experience(self, directory, experience, avg_score):
        experience["after"] = avg_score
        experience["succeed"] = bool(avg_score > experience["before"])

        write_json_file(os.path.join(directory, "experience.json"), experience, encoding="utf-8", indent=4)

    def format_pooled_experience(
        self,
        representative_id,
        family_group,
        all_experience
    ):
        """
        格式化来自整个家族的经验，采用“锚点->原则->示例”的三层结构。
        """
        # 1. 分离代表自身的经验和同伴的经验
        representative_exp = all_experience.get(int(representative_id), {})
        peer_experiences = {mid: all_experience.get(int(mid), {}) for mid in family_group if mid != representative_id}

        # 2. 从整个家族中提取设计模式
        family_success_patterns = Counter()
        family_failure_patterns = Counter()
        all_peer_success_mods = []
        
        for member_id in family_group:
            member_exp = all_experience.get(int(member_id), {})
            for _, succ_data in member_exp.get("success", {}).items():
                patterns = self.experience_analyzer.extract_design_patterns(succ_data['modification'])
                family_success_patterns.update(patterns)
                if member_id != representative_id:
                    all_peer_success_mods.append(f"[来自 Round {member_id} 的参考]: {succ_data['modification']}")
            
            for _, fail_data in member_exp.get("failure", {}).items():
                patterns = self.experience_analyzer.extract_design_patterns(fail_data['modification'])
                family_failure_patterns.update(patterns)

        # 3. 构建格式化字符串
        experience_str = f"You are optimizing workflow Round {representative_id}, which belongs to the family {family_group}.\n"
        
        # --- 层次一：锚点 (Anchor) ---
        experience_str += "\n--- Level 1: Direct Experience of the Current Workflow ---\n"
        if representative_exp.get("success") or representative_exp.get("failure"):
            if representative_exp.get("failure"):
                experience_str += "Directly prohibited modifications (led to failures for this specific workflow):\n"
                for _, value in representative_exp["failure"].items():
                    experience_str += f"- {value['modification']}\n"
            if representative_exp.get("success"):
                experience_str += "Directly successful modifications for this workflow:\n"
                for _, value in representative_exp["success"].items():
                    experience_str += f"- {value['modification']}\n"
        else:
            experience_str += "This workflow has no direct modification history.\n"

        # --- 层次二：设计原则 (Design Principles) ---
        experience_str += "\n--- Level 2: Proven Design Principles from the Whole Family ---\n"
        if family_success_patterns:
            experience_str += "Proven successful design patterns for this family include:\n"
            for pattern, count in family_success_patterns.most_common(3):
                experience_str += f"- {pattern} (seen {count} times)\n"
        if family_failure_patterns:
            experience_str += "Commonly failed design patterns for this family to avoid:\n"
            for pattern, count in family_failure_patterns.most_common(3):
                experience_str += f"- {pattern} (seen {count} times)\n"
        if not family_success_patterns and not family_failure_patterns:
            experience_str += "No strong design patterns have emerged from the family's history yet.\n"
        
        # --- 层次三：参考示例 (Reference Examples) ---
        experience_str += "\n--- Level 3: Inspirational Examples from Peer Workflows ---\n"
        if all_peer_success_mods:
            experience_str += "Consider how to apply the principles from these successful modifications on similar workflows:\n"
            # 只选择一两个最有代表性的例子，避免信息过载
            for example in np.random.choice(all_peer_success_mods, size=min(2, len(all_peer_success_mods)), replace=False):
                experience_str += f"- {example}\n"
        else:
            experience_str += "No successful examples from peer workflows are available.\n"
            
        experience_str += "\nNote: Your main goal is to apply the successful DESIGN PRINCIPLES to the current workflow's code, using its direct experience as the primary guide and peer examples as inspiration."
        return experience_str