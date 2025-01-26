import json
import random
from pathlib import Path
from typing import List, Dict, Any

class DataLoader:
    def __init__(self, gsm8k_path: str, hotpot_path: str):
        self.gsm8k_path = Path(gsm8k_path)
        self.hotpot_path = Path(hotpot_path)
        
    def load_gsm8k_data(self) -> List[Dict[str, str]]:
        data = []
        with open(self.gsm8k_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    'question': item['question'],
                    'answer': item['answer']
                })
        return data
    
    def load_hotpot_data(self) -> List[Dict[str, Any]]:
        with open(self.hotpot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def sample_data(self, data: List[Dict], n: int = 100) -> List[Dict]:
        if len(data) < n:
            raise ValueError(f"Cannot sample {n} items from dataset of size {len(data)}")
        return random.sample(data, n)
    
    def save_sampled_data(self, data: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def process(self):
        gsm8k_data = self.load_gsm8k_data()
        sampled_gsm8k = self.sample_data(gsm8k_data)
        self.save_sampled_data(sampled_gsm8k, 'data/input/gsm8k.json')

        hotpot_data = self.load_hotpot_data()
        sampled_hotpot = self.sample_data(hotpot_data)
        self.save_sampled_data(sampled_hotpot, 'data/input/hotpot.json')

if __name__ == "__main__":
    loader = DataLoader('data/input/raw_gsm8k.jsonl', 'data/input/raw_hotpot.json')
    loader.process()