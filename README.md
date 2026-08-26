# CNN Ablation Study — FashionMNIST
 
Étude d'ablation isolant, un facteur à la fois, l'effet de trois techniques
classiques en Computer Vision : les convolutions **1x1 (bottleneck)**, la
**Batch Normalization**, et les **skip connections (résiduelles)**.
 
## Pourquoi ce projet
 
Comprendre *pourquoi* ces techniques fonctionnent en les lisant est une chose ;
les voir agir sur des courbes d'entraînement en est une autre. Ce projet compare
4 architectures strictement identiques en profondeur (même nombre de blocs),
en ne changeant qu'un seul facteur à la fois, pour isoler l'effet de chacun.
 
## Les 4 variantes
 
| Variante | Bloc | BatchNorm | Skip connection |
|---|---|---|---|
| `baseline` | conv 3x3 classique | non | non |
| `bottleneck` | 1x1 → 3x3 → 1x1 | non | non |
| `bottleneck_bn` | 1x1 → 3x3 → 1x1 | oui | non |
| `bottleneck_bn_skip` | 1x1 → 3x3 → 1x1 | oui | oui |
 
Architecture commune : `stem (conv 3x3) → N blocs → Global Average Pooling → FC`.
 
## Hypothèses testées
 
- **Bottleneck (1x1)** : réduit drastiquement le nombre de paramètres pour une
  profondeur identique, en traitant la conv 3x3 coûteuse sur un nombre de
  canaux réduit.
- **BatchNorm** : accélère et stabilise la convergence (courbes de loss plus
  lisses, moins sensibles au learning rate).
- **Skip connections** : permettent d'empiler plus de blocs sans dégradation
  de l'apprentissage, en offrant un chemin direct au gradient lors de la
  backpropagation.
## Résultats
 
Sur FashionMNIST (`channels=32`, `num_blocks=6`, `epochs=10`) :
 
| Variante | Paramètres | Best Val Acc | Test Acc | Temps (s) |
|---|---|---|---|---|
| baseline | ~111 600 | _à compléter_ | _à compléter_ | _à compléter_ |
| bottleneck | ~7 500 | _à compléter_ | _à compléter_ | _à compléter_ |
| bottleneck_bn | ~8 000 | _à compléter_ | _à compléter_ | _à compléter_ |
| bottleneck_bn_skip | ~8 000 | _à compléter_ | _à compléter_ | _à compléter_ |
 
Le bottleneck réduit le nombre de paramètres d'un facteur ~15 à profondeur
égale — le premier résultat est visible avant même l'entraînement.
 
Courbes générées par `plot_results.py` :
- `results/train_loss.png`, `results/val_loss.png`
- `results/train_acc.png`, `results/val_acc.png`
- `results/test_accuracy.png`
## Limites
 
- FashionMNIST est un dataset simple (28x28, niveaux de gris) : les écarts
  entre variantes peuvent être moins marqués que sur un dataset plus complexe
  (ex: CIFAR-10, ImageNet), notamment pour l'effet des skip connections, dont
  le bénéfice est surtout visible sur des réseaux très profonds.
- Un seul run par variante (pas de moyenne sur plusieurs seeds) : les écarts
  de quelques dixièmes de point ne sont pas statistiquement significatifs.
- `num_blocks=6` reste une profondeur modeste ; l'intérêt des skip
  connections serait plus flagrant avec 20+ blocs.
## Utilisation
 
```bash
pip install -r requirements.txt
 
# Lance l'entraînement des 4 variantes
python run_experiments.py --epochs 10 --batch-size 128
 
# Génère les courbes comparatives
python plot_results.py --results-dir ./results
```
 
## Structure
 
```
models.py            # architectures : baseline, bottleneck, +BN, +skip
train.py              # boucle d'entraînement générique + logging
run_experiments.py    # lance les 4 variantes, sauvegarde les résultats
plot_results.py       # génère les courbes comparatives
requirements.txt
```
 