# %%
import pandas as pd
import numpy as np
import os
os.chdir('/data/comics-lijx/jupyter_projects/codon/datas')
papers = pd.read_csv('/data/comics-ljx/archived/vaccine_design/scopus.20240421.csv')
bicod = pd.read_csv(
    '/data/comics-ljx/archived/vaccine_design/CoCoPUTs/Refseq_Bicod_genomic_to_use.tsv', sep='\t')
dinuc = pd.read_csv(
    '/data/comics-ljx/archived/vaccine_design/CoCoPUTs/o537-Refseq_Dinuc.tsv', sep='\t')
junc_dinuc = pd.read_csv(
    '/data/comics-ljx/archived/vaccine_design/CoCoPUTs/o537-Refseq_JuncDinuc.tsv', sep='\t')
cod = pd.read_csv(
    '/data/comics-ljx/archived/vaccine_design/CoCoPUTs/o537-Refseq_species.tsv', sep='\t')
dinuc = dinuc[dinuc.Assembly.isin(bicod.Assembly)]
junc_dinuc = junc_dinuc[junc_dinuc.Assembly.isin(dinuc.Assembly)]
cod = cod[cod.Assembly.isin(junc_dinuc.Assembly)]
bicod.set_index('Name', inplace=True)

# %%
# import polars as pl
# import pandas as pd
# bicod = pl.read_csv(
#     '/data/comics-ljx/archived/vaccine_design/CoCoPUTs/o537-Refseq_Bicod.tsv', separator='\t').to_pandas()
# species = pd.read_csv('/data/comics-ljx/archived/vaccine_design/species.csv')
# # bicod


# def in_species(x):
#     for y in list(species['学名']):
#         if y in x or x in y:
#             return True
#     return False


# bicod[bicod.Species.apply(in_species) & (
# bicod.Organelle ==
# 'genomic')].drop_duplicates('Taxid').to_csv('/data/comics-ljx/archived/vaccine_design/CoCoPUTs/Refseq_Bicod_genomic_to_use.tsv',
# sep='\t', index=False)

# %%
# name_assembly =
assembly_dict = bicod['Assembly'].to_dict()
assembly_dict = {value: key for key, value in assembly_dict.items()}
cod['Name'] = cod.Assembly.map(assembly_dict)
junc_dinuc['Name'] = junc_dinuc.Assembly.map(assembly_dict)
dinuc['Name'] = dinuc.Assembly.map(assembly_dict)
cod.set_index('Name', inplace=True)
junc_dinuc.set_index('Name', inplace=True)
dinuc.set_index('Name', inplace=True)
cod.sort_index(inplace=True)
junc_dinuc.sort_index(inplace=True)
dinuc.sort_index(inplace=True)
bicod.sort_index(inplace=True)
GC = cod['GC%'].to_dict()

# %%
codon_table = {
    'GCT': 'A',
    'GCC': 'A',
    'GCA': 'A',
    'GCG': 'A',
    'CGT': 'R',
    'CGC': 'R',
    'CGA': 'R',
    'CGG': 'R',
    'AGA': 'R',
    'AGG': 'R',
    'TCT': 'S',
    'TCC': 'S',
    'TCA': 'S',
    'TCG': 'S',
    'AGT': 'S',
    'AGC': 'S',
    'ATT': 'I',
    'ATC': 'I',
    'ATA': 'I',
    'TTA': 'L',
    'TTG': 'L',
    'CTT': 'L',
    'CTC': 'L',
    'CTA': 'L',
    'CTG': 'L',
    'GGT': 'G',
    'GGC': 'G',
    'GGA': 'G',
    'GGG': 'G',
    'GTT': 'V',
    'GTC': 'V',
    'GTA': 'V',
    'GTG': 'V',
    'ACT': 'T',
    'ACC': 'T',
    'ACA': 'T',
    'ACG': 'T',
    'CCT': 'P',
    'CCC': 'P',
    'CCA': 'P',
    'CCG': 'P',
    'AAT': 'N',
    'AAC': 'N',
    'GAT': 'D',
    'GAC': 'D',
    'TGT': 'C',
    'TGC': 'C',
    'CAA': 'Q',
    'CAG': 'Q',
    'GAA': 'E',
    'GAG': 'E',
    'CAT': 'H',
    'CAC': 'H',
    'AAA': 'K',
    'AAG': 'K',
    'TTT': 'F',
    'TTC': 'F',
    'TAT': 'Y',
    'TAC': 'Y',
    'ATG': 'M',
    'TGG': 'W',
    'TAG': 'STOP',
    'TGA': 'STOP',
    'TAA': 'STOP'}
dict_aa = {'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
           'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'I': ['ATT', 'ATC', 'ATA'],
           'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
           'V': ['GTT', 'GTC', 'GTA', 'GTG'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
           'N': ['AAT', 'AAC'], 'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'], 'Q': ['CAA', 'CAG'], 'E': ['GAA', 'GAG'],
           'H': ['CAT', 'CAC'], 'K': ['AAA', 'AAG'], 'F': ['TTT', 'TTC'], 'Y': ['TAT', 'TAC'], 'M': ['ATG'],
           'W': ['TGG'], 'STOP': ['TAG', 'TGA', 'TAA']}

# %%
bicod_df = bicod.iloc[:, 8:]
bicod_df = bicod_df.astype(np.float64).div(
    bicod['# Codon Pairs'].astype(np.float64), axis=0)
bicod_df.columns = bicod_df.columns.str.upper()
# bicod_df
cod_df = cod.iloc[:, 11:]
cod_df = cod_df.astype(np.float64).div(
    cod['# Codons'].astype(np.float64), axis=0)

# %%
# Section 1: Initialize Dictionaries
dict_bicod = {}  # Dictionary to store bi-codon relationships
dict_biaa = {}   # Dictionary to store bi-amino acid relationships

# Section 2: Generate Bi-amino acid and Bi-codon Mappings
for a1 in dict_aa:
    for a2 in dict_aa:
        # Skip if first amino acid is a stop codon
        if a1 == 'STOP':
            continue
        # Generate all possible combinations of codons for each amino acid pair
        for c1 in dict_aa[a1]:
            for c2 in dict_aa[a2]:
                biaa = a1 + a2    # Combine two amino acids
                bicod = c1 + c2   # Combine their corresponding codons
                # Initialize list for new bi-amino acid if not exist
                if biaa not in dict_biaa:
                    dict_biaa[biaa] = []
                # Add bi-codon to corresponding bi-amino acid list
                dict_biaa[biaa].append(bicod)

# Section 3: Build Bi-codon Relationships
for val_list in dict_biaa.values():
    for val in val_list:
        # Initialize list for new bi-codon if not exist
        if val not in dict_bicod:
            dict_bicod[val] = []
        # Extend bi-codon list with related bi-codons
        dict_bicod[val].extend(val_list)

# %%
# Section 4: Create Amino Acid Frequency DataFrame
aa_df = pd.DataFrame(columns=dict_aa.keys(), index=cod_df.index)
aa_df.fillna(0, inplace=True)  # Fill NA values with 0
# Sum up codon frequencies for each amino acid
for aa in dict_aa:
    for codon in dict_aa[aa]:
        if codon in cod_df.columns:
            aa_df[aa] += cod_df[codon]

# %%
# Section 5: Create Bi-amino Acid Frequency DataFrame
biaa_df = pd.DataFrame(columns=dict_biaa.keys(), index=cod_df.index)
biaa_df.fillna(0, inplace=True)  # Fill NA values with 0
# Sum up bi-codon frequencies for each bi-amino acid
for biaa in dict_biaa:
    for codon in dict_biaa[biaa]:
        if codon in bicod_df.columns:
            biaa_df[biaa] += bicod_df[codon]

# %%
# Section 6: Calculate Codon Pair Scores (CPS)
cps_df = pd.DataFrame(columns=dict_bicod.keys(), index=cod_df.index)
cps_df.fillna(0, inplace=True)  # Fill NA values with 0
# Calculate CPS for each bi-codon
for bicod in dict_bicod.keys():
    if bicod in bicod_df.columns:
        # CPS formula: ln(observed_frequency / expected_frequency)
        # where expected_frequency = (f1 * f2)/(a1 * a2) * f12
        # f1, f2: individual codon frequencies
        # a1, a2: individual amino acid frequencies
        # f12: bi-amino acid frequency
        cps_df[bicod] = np.log(
            (bicod_df[bicod]) /
            (((cod_df[bicod[0:3]] * cod_df[bicod[3:]]) /
              (aa_df[codon_table[bicod[0:3]]] * aa_df[codon_table[bicod[3:]]])) *
             biaa_df[codon_table[bicod[0:3]] + codon_table[bicod[3:]]])
        )

# %%
# Create a new DataFrame to store ln(w) values for CAI calculation
# Using the same structure as cod_df (same columns and index)
cai_lnw_df = pd.DataFrame(columns=cod_df.columns, index=cod_df.index)

# Iterate through each amino acid
for aa in dict_aa.keys():
    # Create temporary DataFrame with codons for current amino acid only
    cod_df_tmp = cod_df[dict_aa[aa]].copy()

    # Find maximum codon frequency for each row (reference set)
    # This represents the frequency of the most used codon for this amino acid
    max_x = cod_df_tmp.max(axis=1)

    # Calculate ln(w) for each codon encoding this amino acid
    # w = codon_frequency / max_codon_frequency
    # ln(w) is stored for later CAI calculations
    for cod in dict_aa[aa]:
        cai_lnw_df[cod] = np.log(cod_df[cod] / max_x)
