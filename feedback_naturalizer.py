"""
Feedback Naturalizer - NLP Scoring Mechanism

Detects repetitive VLM responses using cosine similarity
and returns natural variations to avoid robotic repetition.

Usage:
    from feedback_system_simple import FeedbackNaturalizer

    naturalizer = FeedbackNaturalizer()

    # Each time you get a response from your VLM:
    result = naturalizer.process("Keep your back straight")
    print(result['display'])  # Show this to user
"""

import random
import time
import numpy as np
from sentence_transformers import SentenceTransformer


class FeedbackNaturalizer:
    """
    Model-agnostic feedback naturalizer.
    Works with any VLM (MobileVideoGPT, LLaVA, Gemini, etc.)
    """

    PHRASES = {
        1: [
            "Still the same issue",
            "Same mistake again",
            "This error is still there",
            "You're still making this mistake",
            "Still seeing this problem",
        ],
        2: [
            "Still there - {short}",
            "Remember: {short}",
            "Keep focusing on: {short}",
            "This hasn't improved yet - {short}",
            "Again: {short}",
        ],
        3: [
            "Keep working on this",
            "You need more practice on this",
            "Focus on correcting this",
            "Still needs improvement",
            "Let's keep trying to fix this",
        ]
    }

    def __init__(self, threshold=0.70, history_size=10, auto_clear=None):
        """
        Args:
            threshold: Similarity threshold (0-1). Lower = more sensitive. Default: 0.70
            history_size: Max responses to remember. Default: 10
            auto_clear: Auto-reset after N seconds of inactivity. Default: None (disabled)
        """
        self.threshold = threshold
        self.history_size = history_size
        self.auto_clear = auto_clear
        self.last_time = time.time()
        self.history = []

        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Ready!")

    def process(self, feedback: str) -> dict:
        """
        Process VLM output and return naturalized response.

        Args:
            feedback: Raw text from your VLM

        Returns:
            dict with 'display' (show to user), 'original', 'is_repeat', 'similarity', 'repeat_count'
        """
        # Auto-clear if inactive
        if self.auto_clear and (time.time() - self.last_time > self.auto_clear):
            self.reset()
        self.last_time = time.time()

        # Get embedding
        emb = self.model.encode(feedback)

        # Check for similar previous feedback
        for i, (_, prev_emb, count) in enumerate(self.history[-5:]):
            sim = np.dot(emb, prev_emb) / (np.linalg.norm(emb) * np.linalg.norm(prev_emb))

            if sim >= self.threshold:
                new_count = count + 1
                idx = len(self.history) - 5 + i
                if idx >= 0:
                    self.history[idx] = (feedback, emb, new_count)

                # Get variation phrase
                level = min(new_count, 3)
                phrase = random.choice(self.PHRASES[level])
                short = feedback.split('.')[0][:50]
                display = phrase.format(short=short) if '{short}' in phrase else phrase

                return {
                    'display': display,
                    'original': feedback,
                    'is_repeat': True,
                    'similarity': float(sim),
                    'repeat_count': new_count
                }

        # New feedback - add to history
        self.history.append((feedback, emb, 0))
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return {
            'display': feedback,
            'original': feedback,
            'is_repeat': False,
            'similarity': 0.0,
            'repeat_count': 0
        }

    def reset(self):
        """Clear history (call when starting new exercise)."""
        self.history = []
        self.last_time = time.time()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Feedback Naturalizer - Test")
    print("=" * 60)

    naturalizer = FeedbackNaturalizer(threshold=0.70)

    vlm_outputs = [
        "Keep your back straight during the squat",
        "Keep your back straight during the squat",
        "Maintain a straight back while squatting",
        "Keep your back straight",
        "Your knees are going too far forward",
        "Your knees are extending past your toes",
    ]

    print("\nSimulating VLM responses:\n")
    for i, output in enumerate(vlm_outputs, 1):
        result = naturalizer.process(output)
        status = f"[Repeat #{result['repeat_count']}, sim: {result['similarity']:.2f}]" if result['is_repeat'] else "[New]"
        print(f"{i}. \"{output}\"")
        print(f"   → \"{result['display']}\" {status}\n")

    print("=" * 60)
