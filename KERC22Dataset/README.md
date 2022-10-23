# KERC22 Dataset

KERC22 dataset consists of total 12289 sentences from 1513 scenes of a Korean TV show named 'Three Brothers'.
For KERC22 challenge the dataset is split into Train and test sets.
Each sample consists of sentence_id, person(speaker), sentence, scene_ID, context(Scene description).

- Train : 7339 (9/1 Release)
- Public_Test: 2566 (9/1 Release)
- Test: 2384  **(10/4 Release)**


Each sentence is labeled with one of the following complex emotion labels: euphoria, dysphoria and neutral

Goal is to correctly predict the socio-behavioral emotional state (euphoria, dysphoria or neutral) of the speaker for each spoken sentence.

### Files
- Training Data: train_data.tsv
- Public Test Data: val_data.tsv
- Private Test Data: test_data.tsv *   **(10/4 Release)**

