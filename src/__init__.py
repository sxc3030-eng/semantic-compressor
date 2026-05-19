"""Semantic Compressor — POC.

Compresse une base de donnees en stockant une recette (.md lisible) + des ancres
(.parquet minimales) au lieu des donnees brutes. Regenere la BD avec une fidelite
statistique elevee.

Modules :
- models           : Schemas Pydantic partages
- profiler         : Profil statistique par colonne
- pattern_detector : Distributions, correlations, dependances fonctionnelles
- anchor_extractor : Separation irreductible / regenerable
- recipe_writer    : Serialisation Markdown de la recette
- reconstructor    : Regeneration deterministe depuis recette + ancres
- validator        : Comparaison original vs reconstruit
- orchestrator     : Pipeline complet + CLI
"""

__version__ = "0.1.0"
