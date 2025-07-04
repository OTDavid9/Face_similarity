from typing import List

def verify_identity(top_k_matches:List):
        # Configuration parameters (tune based on your data)
    THRESHOLD_HIGH = 0.85       # High-confidence match
    THRESHOLD_LOW = 0.65        # Minimum similarity to consider
    MARGIN = 0.05  - 1e-2             # Minimum score gap between #1 and #2
    LIKENESS_THRESHOLD = 0.78   # Visual likeness threshold

     # Unpack top matches
    top_match = top_k_matches[0]
    second_match = top_k_matches[1] if len(top_k_matches) > 1 else None

    diff_first_two = (top_match["score"] - second_match["score"])

    margin_diff = diff_first_two >= MARGIN


    # top_match = {
    #     'id': '22366746848',
    #     'metadata': {'bvn': '22366746848'},
    #     'score': 0.829854786,
    #     'values': []
    # }
    similarity_score = top_match["score"]

    # --- Core Verification Flow ---
    # Case 1: Clear high-confidence match
    if similarity_score  < 0.65:
        response = {
            "status": "LIKELY TO BE A NEW USER Based on low similarity",
            "BVN": top_match["id"],
            "similarity_score": round(similarity_score, 3),
            "margin":diff_first_two

        }
        print(response)
        return response
    

    elif similarity_score >= THRESHOLD_HIGH or margin_diff:
        response = {
            "status": "EXISTING_USER",
            "BVN": top_match["id"],
            "similarity_score": round(similarity_score, 3),
            "margin":diff_first_two

        }
        print(response)
        return response
            
    else:
        response = {
            "status": "LIKELY TO BE A NEW USER",
            "BVN": top_match["id"],
            "similarity_score": round(similarity_score, 3)}
        print({"status":"LIKELY TO BE A NEW USER", "margin":diff_first_two})
        return response


# # Sample input (from your example)
# top_k_matches  = [
#     {"id": "22366746848", "score": 0.8899, "bvn": "22366746848"},
#     {"id": "22174864493", "score": 0.8853, "bvn": "22174864493"},
#     {"id": "22169192587", "score": 0.6902, "bvn": "22169192587"},
#     {"id": "22445253966", "score": 0.6841, "bvn": "22445253966"}
# ]

# verify_identity(top_k_matches)

#########################
