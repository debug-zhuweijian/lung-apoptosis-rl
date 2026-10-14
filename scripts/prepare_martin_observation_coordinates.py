"""Exclusive CPU coordinate preparation; never train or execute a model."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
from lung.data.martin_observation_coordinates import prepare

if __name__ == '__main__':
    m = prepare()
    print({'folds':m['fold_count'],'genes':m['gene_count'],'observed':m['observed_genes'],
           'unique_fit_counts':[f['unique_fit_count'] for f in m['folds']], 'model_forwards':0})
