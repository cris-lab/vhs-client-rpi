from typing import List

class Detection:
    
    def __init__(
        self, 
        category_id: str, 
        label: str, 
        score: float, 
        bbox: List[int]
    ):
        
        self.category_id   = category_id
        self.label      = label
        self.score = score
        self.bbox       = bbox
    
    def to_dict(self):
        return {
            'category_id': self.category_id,
            'label': self.label,
            'score': self.score,
            'bbox': self.bbox.tolist()
        }
    
    def __str__(self):
        return f"category_id: {self.category_id}, label: {self.label}, score: {self.score}, bbox: {self.bbox}"