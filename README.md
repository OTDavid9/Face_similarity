1. New face embedding → query vector DB
2. Get top 5 matches with similarity scores

3. If top-1 similarity ≥ 0.88 → MATCH (high confidence)
4. Else if (top-1 ≥ 0.82) AND (top1 - top2 > 0.05) → MATCH (moderate confidence)
5. Else → UNCERTAIN → Ask for another image or supporting ID# Face_similarity
