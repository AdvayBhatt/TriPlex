"""Synthetic demonstration only: new families every season, two breeding pools."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260912)
    markers = [f'SNP{i}' for i in range(64)]
    for c in (1, 2):
        folder = root / f'genotypes/C{c}'
        folder.mkdir(parents=True, exist_ok=True)
        beta = rng.normal(0, 1.5, len(markers))
        rows = []
        for year in range(2001, 2009):
            for family in range(4):
                pop = (year - 2001) * 4 + family + 1
                dna = rng.integers(-1, 2, (24, len(markers))).astype(float)
                values = dna @ beta + rng.normal(0, 3, len(dna))
                dna[rng.random(dna.shape) < .01] = np.nan
                pd.DataFrame(dna, index=[f'{i:011d}' for i in range(1, 25)], columns=markers).to_csv(
                    folder / f'C{c}.{pop}_Imputed.csv')
                for i in range(1, 25):
                    for site in range(3):
                        rows.append(dict(LINE_UNIQUE_ID=f'C{c}.{pop}.{i}', YEAR_x=year,
                                         LOC=f'S{site}', CROSS=f'{family + 1}/{family + 10}',
                                         YLD_BE=np.nan if year == 2008 and i == 1 else
                                         140 + site * 15 + (year - 2001) * 2 + values[i - 1] + rng.normal(0, 6),
                                         MST=rng.normal(20, 2)))
        pd.DataFrame(rows).to_csv(root / f'C{c}_Phenotype_Data_V2.csv', index=False)
