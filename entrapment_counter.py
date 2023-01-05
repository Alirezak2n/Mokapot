import pandas as pd
import logging


def proteins_variety(protein_file=None):
    if not protein_file:
        protein_file = './fasta_proteins.csv'
    fasta_protein = pd.read_csv(protein_file, index_col='Protein')
    fasta_protein_dict = fasta_protein.T.to_dict()

    return fasta_protein_dict


def protein_checker(protein, protein_dict):
    if "\t" in protein:
        pros = []
        for key in protein.split('\t'):
            if protein_dict.get(key):
                pros.append(protein_dict.get(key)['Type'])
            else:
                pros.append('Contaminant')
        pros = list(set(pros))
        if len(pros) == 1:
            return pros.pop()
        else:
            return sorted(pros)
    else:
        if protein_dict.get(protein):
            return protein_dict.get(protein)['Type']
        else:
            return 'Contaminant'


def protein_encoder(protein):
    encod = []
    if not isinstance(protein, list):
        protein = [protein]
    for prot in protein:
        if prot == "HUMAN":
            encod.append('1')
        elif prot == "Contaminant":
            encod.append('3')
        else:
            encod.append('2')
    encod = list(set(encod))
    return ('').join(sorted(encod))

def entapments_peptide_exporter(psms_table=None, fdr=0.01):
    if psms_table is None:
        psms_table = pd.read_table('./saved_models/20221221/mlp_0611.mokapot.psms.txt')
    else:
        psms_table = pd.read_table(psms_table)
    psms_table = psms_table[psms_table['mokapot q-value'] <= fdr]
    protein_dict = proteins_variety()
    psms_table['Protein_type'] = psms_table.apply(lambda x: protein_checker(x.Proteins, protein_dict), axis=1)
    psms_table['Protein_encod'] = psms_table.apply(lambda x: protein_encoder(x.Protein_type), axis=1)
    psms_table_entapments = psms_table[psms_table['Protein_encod'] == '2']
    psms_table_entapments = psms_table_entapments['Peptide'].tolist()
    return psms_table_entapments

def id_counter(psms_table):
    psms_table = pd.read_table(psms_table)
    length_1fdr = len(psms_table[psms_table['mokapot q-value']<=0.01])
    q_value_serie=psms_table['mokapot q-value']
    psms_index=psms_table.index
    return q_value_serie, psms_index, length_1fdr

def main(psms_table=None, fdr=0.01):
    if psms_table is None:
        psms_table = pd.read_table("./mokapot_default.mokapot.psms.txt")
    else:
        psms_table = pd.read_table(psms_table)
    psms_table = psms_table[psms_table['mokapot q-value'] <= fdr]
    protein_dict = proteins_variety()
    psms_table['Protein_type'] = psms_table.apply(lambda x: protein_checker(x.Proteins, protein_dict), axis=1)
    psms_table['Protein_encod'] = psms_table.apply(lambda x: protein_encoder(x.Protein_type), axis=1)
    entraps_Human = len(psms_table[psms_table['Protein_encod'] == '2']) / len(
        psms_table[psms_table['Protein_encod'] == '1']) * 100
    values = psms_table['Protein_encod'].value_counts().to_dict()
    logging.info("1=Human, 2=Entrapment, 3=Contaminant \n %s", values)
    logging.info(f'Entrapment ratio to Human is: {entraps_Human: .5f}%')
    # print("1=Human, 2=Entrapment, 3=Contaminant \n ", values)
    # print(f'Entrapment ratio to Human is: {entraps_Human: .5f}%')

    return entraps_Human


if __name__ == '__main__':
    main()
