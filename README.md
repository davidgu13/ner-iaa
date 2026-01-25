# ner-iaa

A helper module for measuring inter-annotator agreement for Named Entity Recognition, specifically for Doccano labels.

### Features
1. Metrics: mutual F1 score, mutual Cohen's Kappa score
1. Supported both ignoring and accounting for "O" label occurrences

1. Entity types are statically defined, not dynamically inferred

### Usage
```buildoutcfg
ner_iaa = NERInterAnnotatorAgreementCharacterLevel()
measures = ner_iaa(text, doccano_labels1, doccano_labels2)
print(measures)
```
Use the class `NERInterAnnotatorAgreement` for space-tokenized text, and `NERInterAnnotatorAgreementCharacterLevel` for tokenizer-agnostic evaluation.

### Limitations
1. Nested NER labels are not supported
1. Partial overlap doesn't affect the scores
