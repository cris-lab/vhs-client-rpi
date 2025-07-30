import numpy as np
import json
from datetime import datetime
from typing import Tuple, List

class HeatMap:
    def __init__(
        self, 
        model,
        frame_size: Tuple[int, int], 
        grid_size: Tuple[int, int] = (20, 20), 
        decay_factor: float = 0.9
    ):
        self.model = model
        self.frame_height, self.frame_width = frame_size
        self.grid_rows, self.grid_cols = grid_size
        self.decay_factor = decay_factor

        self.heatmap = np.zeros(grid_size, dtype=np.float32)
        self.cell_height = self.frame_height / self.grid_rows
        self.cell_width = self.frame_width / self.grid_cols
        
        self.counter = 0
        self.max = 100

    def analyze(self, frame: np.ndarray):
        """
        Ejecuta inferencia y actualiza el heatmap con personas detectadas.
        """
        
        self.counter += 1
        if self.counter >= self.max:
            self.counter = 0
        
        if self.counter > 0:
            print("[INFO] Skipping heatmap analysis due to counter limit.")
            return

        print("[INFO] 🔥 Analizando frame para heatmap...")
        result = self.model(frame)

        for res in result.results:  # asumir res['bbox'] y res['label']
            if res.get('label', '').lower() != 'person':
                continue

            x1, y1, x2, y2 = map(int, res['bbox'])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            row = int(cy / self.cell_height)
            col = int(cx / self.cell_width)

            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                self.heatmap[row, col] += 1

    def decay(self):
        self.heatmap *= self.decay_factor

    def summarize(self, threshold: float = 5.0) -> dict:
        flat = self.heatmap.ravel()
        top_indices = np.argsort(flat)[::-1][:3]
        top_zones = [
            {
                "pos": [int(i // self.grid_cols), int(i % self.grid_cols)],
                "value": float(flat[i])
            }
            for i in top_indices
        ]
        avg = float(np.mean(self.heatmap))
        hot_ratio = float(np.sum(self.heatmap > threshold) / self.heatmap.size)

        return {
            "timestamp": datetime.now().isoformat(),
            "top_zones": top_zones,
            "avg_activity": avg,
            "hot_zone_ratio": hot_ratio
        }

    def save_summary(self, path: str, threshold: float = 5.0):
        summary = self.summarize(threshold)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        self.heatmap.fill(0)

    def get_map(self) -> np.ndarray:
        return self.heatmap.copy()
