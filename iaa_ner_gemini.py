from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import f1_score, cohen_kappa_score

def calculate_ner_agreement(annotator1_tags, annotator2_tags):
    """
    Calculates Entity-level F1 and Cohen's Kappa for BIO tags.
    Includes 'O' as a category.
    """

    # 1. Extract entities (spans) from the BIO sequences
    # seqeval returns list of ('entity_type', start_index, end_index)
    entities1 = get_entities(annotator1_tags)
    entities2 = get_entities(annotator2_tags)

    # 2. Map tokens to their entity types
    # To include 'O', we create a flattened list of labels for every token
    # but we ensure the labels reflect the "Entity" they belong to.
    y_true = annotator1_tags
    y_pred = annotator2_tags

    # Get unique entity types (e.g., ['PER', 'LOC', 'ORG', 'O'])
    unique_entities = sorted(list(set([e[0] for e in entities1 + entities2] + ['O'])))

    results = {}

    for entity in unique_entities:
        # Binary classification for this specific entity type vs everything else
        # This allows us to calculate F1 and Kappa per entity
        y_true_binary = [1 if tag.endswith(entity) or (tag == 'O' and entity == 'O') else 0 for tag in y_true]
        y_pred_binary = [1 if tag.endswith(entity) or (tag == 'O' and entity == 'O') else 0 for tag in y_pred]

        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

        results[entity] = {"F1": round(f1, 4), "Kappa": round(kappa, 4)}

    return results

# --- Example Usage ---
# Labels must be word-level BIO format
ann1 = ["B-PER", "I-PER", "O", "B-LOC", "O"]
ann2 = ["B-PER", "I-PER", "O", "O", "B-LOC"]

scores = calculate_ner_agreement(ann1, ann2)

print(f"{'Entity':<10} | {'F1 Score':<10} | {'Kappa':<10}")
print("-" * 35)
for ent, metrics in scores.items():
    print(f"{ent:<10} | {metrics['F1']:<10} | {metrics['Kappa']:<10}")