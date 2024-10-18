# Exploration des Corrélations & Modèles Prédictifs des Niveaux de Vitamine D

## Overview
Ce projet vise à explorer les corrélations entre divers tests cliniques et les niveaux de vitamine D, ainsi qu'à développer des modèles prédictifs pour estimer ces niveaux. L'outil est conçu pour aider les professionnels de la santé à prendre des décisions éclairées concernant les soins aux patients tout en réduisant les tests inutiles et les coûts associés.

## Features
- **Analyse des Corrélations** : Utilisation des méthodes de Spearman et Kendall pour analyser les relations entre les niveaux de vitamine D et divers indicateurs de santé.
- **Modélisation Prédictive** : Application de techniques de régression pour estimer les niveaux de vitamine D en fonction d'autres tests cliniques.
- **Application Web Conviviale** : Développée avec Flask, permettant aux patients d'entrer leurs données cliniques et de recevoir des prédictions sur leurs niveaux de vitamine D.
- **Visualisation des Données** : Utilisation de Power BI pour visualiser les résultats et faciliter la communication des findings.

## Data
Le jeu de données utilisé dans ce projet est stocké dans un fichier Excel nommé **"Vitamine_D_modified.xlsx"**, qui comprend divers tests cliniques et les niveaux de vitamine D correspondants.

### Data Variables
- **Variables des Tests Cliniques** : Divers tests utilisés pour évaluer les indicateurs de santé pertinents aux niveaux de vitamine D.
- **Niveau de Vitamine D** : La variable cible pour la prédiction.

## Installation et Utilisation
### Prérequis
- **Python** (version 3.6 ou supérieure)
- Packages Python requis : `Flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`...
